import numpy as np
import cv2
import os

from mpl_toolkits import mplot3d
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from fast_detector import FastDetector
from heat_map import Event, HeatMap


path = "/home/palinauskas/Documents/corner_events_py/event_sample"


def read_packages(path):
    keys = ["x", "y", "t", "p"]
    total_data = {k: np.array([]) for k in keys}

    files = os.listdir(path)
    files.sort()

    count = 0
    for f in files[:100]:
        
        if f.endswith(".npz"):
            count += 1
            # print(f)
            data = np.load(os.path.join(path, f))
            for k in keys:
                # print(data[k].size)
                total_data[k] = np.concatenate((total_data[k], data[k]), axis=None)
        
        # if count > 5:
        #     break

    min_time, max_time = min(total_data["t"]), max(total_data["t"])
    min_x, max_x = min(total_data["x"]), max(total_data["x"])
    min_y, max_y = min(total_data["y"]), max(total_data["y"])
    H, W = 64, 64
    xs = (max_x - min_x) * 1e-2
    ys = (max_y - min_y) * 1e-2
    ts =  (max_time - min_time) * 1e-7
    print(ts, xs, ys)


    events = []
    for i in range(total_data["x"].size):
        p = False
        if total_data["p"][i] == 1:
            p = True
        events.append(Event(total_data["x"][i], total_data["y"][i], p, total_data["t"][i]))

    return events

events = read_packages(path)

f, axarr = plt.subplots(3,2)

fd = FastDetector()

t = -1

img = np.zeros((64,64))
for e in events:
    if e.t - t > 0.001:
        t = e.t
        img *= 0.99
    # print(e.x, e.y, e.polarity)
    answer = fd.isFeature(e)
    # print(answer)
    img[e.x, e.y] = answer

# plt.imshow(img)
# plt.show()

fd2 = FastDetector()

hm = HeatMap(64,64)
for e in events:
    
    if fd2.isFeature(e):
        hm.addEvent(e)


axarr[0,0].imshow(img)
axarr[0,1].imshow(hm.getMap())
axarr[1,0].imshow(hm.getCornerMap())
axarr[1,1].imshow(hm.getCornerClusters()[0])
axarr[2,0].imshow(hm.getCorners())
axarr[2,1].imshow(hm.getCorners())
plt.show()