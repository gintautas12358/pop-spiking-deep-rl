from modulefinder import packagePathMap
import numpy as np

from distict_queue import DistinctQueue

class HarrisDetector:

    def __init__(self):
        # parameters
        self.queue_size_ = 25 #25
        self.window_size_ = 4
        self.kernel_size_ = 5
        self.harris_threshold_ = 8.0
        self.sensor_width_ = 64
        self.sensor_height_ = 64

        self.last_score_ = -999

        self.queue = DistinctQueue(self.window_size_, self.queue_size_, True)

        self.Dx = np.zeros((self.kernel_size_))
        self.Sx = np.zeros((self.kernel_size_))

        for i in range(self.kernel_size_):
            self.Sx[i] = self.factorial(self.kernel_size_ - 1)/(self.factorial(self.kernel_size_ - 1 - i) * self.factorial(i))
            self.Dx[i] = self.pasc(i, self.kernel_size_-2) - self.pasc(i-1, self.kernel_size_-2)

        self.Gx_ = np.outer(self.Sx, self.Dx)
        self.Gx_ /= self.Gx_.max()

        sigma = 1
        A = 1./(2.*np.pi*sigma*sigma)
        l2 = int( (2*self.window_size_+2-self.kernel_size_)/2 )
        size = 2 * l2 + 1
        self.h_ = np.zeros((size, size))

        for x in range(-l2, l2+1):
            for y in range(-l2, l2+1):
                a = (x*x+y*y)/(2*sigma*sigma)
                h_xy = A * np.exp(-a)
                self.h_[l2+x, l2+y] = h_xy

        self.h_ /= self.h_.sum()

    def isFeature(self, e):
        # update queues
        self.queue.newEvent(e.x, e.y, e.polarity)

        # check if queue is full
        score = self.harris_threshold_ - 10
        if self.queue.isFull(e.x, e.y, e.polarity):
            # check if current event is a feature
            score = self._getHarrisScore(e.x, e.y, e.polarity)
            self.last_score_ = score

        return (score > self.harris_threshold_)

    def getLastScore(self):
        return self.last_score_

    def _updateQueue(self):
        pass

    def _getHarrisScore(self, img_x, img_y, polarity):
        if img_x < self.window_size_ or img_x > self.sensor_width_ - self.window_size_ or \
            img_y < self.window_size_ or img_y > self.sensor_height_ - self.window_size_:
            return self.harris_threshold_ - 10

        local_frame = self.queue.getPatch(img_x, img_y, polarity)
        l = 2*self.window_size_+2-self.kernel_size_
        dx = np.zeros((l, l))
        dy = np.zeros((l, l))

        for x in range(l):
            for y in range(l):
                for kx in range(self.kernel_size_):
                    for ky in range(self.kernel_size_):
                        dx[x, y] += local_frame[x+kx, y+ky]*self.Gx_[kx, ky]
                        dy[x, y] += local_frame[x+kx, y+ky]*self.Gx_[ky, kx]

        a = b = d = 0
        for x in range(l):
            for y in range(l):
                a += self.h_[x, y] * dx[x, y] * dx[x, y]
                b += self.h_[x, y] * dx[x, y] * dy[x, y]
                d += self.h_[x, y] * dy[x, y] * dy[x, y]

        score = a*d-b*b - 0.04*(a+d)*(a+d)

        return score


    def factorial(self, n):
        if n > 1:
            return n * self.factorial(n - 1)
        else:
            return 1

    def pasc(self, k, n):
        if k >= 0 and k <= n:
            return self.factorial(n)/(self.factorial(n-k)*self.factorial(k))
        else:
            return 0