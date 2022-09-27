import numpy as np

from fixed_distict_queue import FixedDistinctQueue

class DistinctQueue:

    def __init__(self, window_size, queue_size, use_polarity) -> None:
        self.window_size_ = window_size
        self.queue_size = queue_size

        self.sensor_width_ = 64
        self.sensor_height_ = 64

        # create queues
        polarities = 1 # 2 or 1
        if use_polarity:
            polarities = 2 # 2 or 1

        num_queue = self.sensor_width_ * self.sensor_height_ * polarities

        self.queues_ = [FixedDistinctQueue(2*window_size+1, queue_size)] * num_queue

    def isFull(self, x, y, pol):
        return self.queues_[self.getIndex(x, y, pol)].isFull()

    def newEvent(self, x, y, pol):
        # update neighboring pixels
        for dx in range(-self.window_size_, self.window_size_+1):
            for dy in range(-self.window_size_, self.window_size_+1):
                # in limits?
                if x + dx < 0 or x + dx >= self.sensor_width_ or y + dy <0 or y + dy >= self.sensor_height_:
                    continue

                # update pixel's queue
                index = self.getIndex(x + dx, y + dy, pol)
                self.queues_[index].addNew(self.window_size_ + dx, self.window_size_ + dy)

    def getPatch(self, x, y, pol):
        return self.queues_[self.getIndex(x, y, pol)].getWindow()

    def getIndex(self, x, y, polarity):
        polarity_offset = 0
        if polarity:
            polarity_offset = self.sensor_width_ * self.sensor_height_
        return int(y * self.sensor_width_ + x + polarity_offset)

    def size(self):
        return len(self.queues_)

    def print(self):
        for q in self.queues_:
            print("#")
            q.print()