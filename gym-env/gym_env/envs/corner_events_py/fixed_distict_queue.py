import numpy as np

class QueueEvent:
    def __init__(self) -> None:
        self.prev = -1
        self.next = -1
        self.x = 0
        self.y = 0

    def __repr__(self) -> str:
        return f"{self.x}, {self.y}, {self.prev}, {self.next}"

class FixedDistinctQueue:
    def __init__(self, window, queue) -> None:
        self.first_ = -1
        self.last_ = -1

        self.queue_max_ = queue
        self.window_ = - np.ones((window, window))
        self.queue_ = []

    def isFull(self):
        return len(self.queue_) >= self.queue_max_

    def addNew(self, x, y):
        # queue full?
        if len(self.queue_) < self.queue_max_:
            if self.window_[x, y] < 0:
                # first element?
                if len(self.queue_) == 0:
                    self.first_ = 0 
                    self.last_ = 0

                    qe = QueueEvent()
                    qe.prev = -1
                    qe.next = -1
                    qe.x = x
                    qe.y = y
                    self.queue_.append(qe)

                    self.window_[x, y] = 0

                else:
                    # add new element
                    qe = QueueEvent()
                    qe.prev = -1
                    qe.next = self.first_
                    qe.x = x
                    qe.y = y
                    self.queue_.append(qe)

                    place = len(self.queue_) - 1
                    self.queue_[self.first_].prev = place
                    self.first_ = place
            else:
                # link neighbors of old event in queue
                place = int(self.window_[x, y])

                if self.queue_[place].next >= 0 and self.queue_[place].prev >= 0:
                    self.queue_[self.queue_[place].prev].next = self.queue_[place].next
                    self.queue_[self.queue_[place].next].prev = self.queue_[place].prev

                # relink first and last
                if place == self.last_:
                    if self.queue_[place].prev >= 0:
                        self.last_ = self.queue_[place].prev
                        self.queue_[self.queue_[place].prev].next = -1

                self.queue_[self.first_].prev = place

                self.queue_[place].prev = -1
                if self.first_ != place:
                    self.queue_[place].next = self.first_
                
                self.first_ = place
        else:
            # is window empty at location
            if self.window_[x, y] < 0:
                # update window
                self.window_[self.queue_[self.last_].x, self.queue_[self.last_].y] = -1
                self.window_[x, y] = self.last_

                # update queue
                self.queue_[self.queue_[self.last_].prev].next = -1
                self.queue_[self.last_].x = x
                self.queue_[self.last_].y = y
                self.queue_[self.last_].next = self.first_
                second_last = self.queue_[self.last_].prev
                self.queue_[self.last_].prev = -1
                self.queue_[self.first_].prev = self.last_
                self.first_ = self.last_
                self.last_ = second_last
            else:
                place = int(self.window_[x, y])
                if place != self.first_:
                    # update window
                    self.window_[x, y] = place

                    # update queue
                    if (self.queue_[place].prev != -1):
                        self.queue_[self.queue_[place].prev].next = self.queue_[place].next
                    
                    if (self.queue_[place].next != -1):
                        self.queue_[self.queue_[place].next].prev = self.queue_[place].prev
                    
                    if (place == self.last_):
                        self.last_ = self.queue_[self.last_].prev
                    
                    self.queue_[place].prev = -1
                    self.queue_[place].next = self.first_
                    self.queue_[self.first_].prev = place

                    self.first_ = place

    def getWindow(self):
        patch = self.window_.copy()
        return np.where(patch < 0, 0, 1)
        
    def print(self):
        for q in self.queue_:
            print("##", q)
