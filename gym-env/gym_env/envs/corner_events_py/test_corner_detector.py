import numpy as np
import cv2
import os

from mpl_toolkits import mplot3d
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from harris_detector import HarrisDetector
from heat_map import Event

    
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

hd = HarrisDetector()

max = -1000

img = np.zeros((64,64))
for e in events:
    # print(e.x, e.y, e.polarity)
    hd.isFeature(e)
    score = hd.getLastScore()
    if score > max:
        max = score
        print(score)
    img[int(e.x), int(e.y)] = score

# e = Event(33.0, 10.0, -1.0)
# for i in range(300):
#     print(hd.isFeature(e))
#     score = hd.getLastScore()
#     print(score)
#     img[int(e.x), int(e.y)] = score

# print(hd.queue.print())

plt.imshow(img)
plt.show()