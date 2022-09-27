import numpy as np

class FastDetector:
    def __init__(self, w, h) -> None:
        self.sensor_width_ = w
        self.sensor_height_ = h


        self.circle3_ = [[0, 3], [1, 3], [2, 2], [3, 1],
            [3, 0], [3, -1], [2, -2], [1, -3],
            [0, -3], [-1, -3], [-2, -2], [-3, -1],
            [-3, 0], [-3, 1], [-2, 2], [-1, 3]]
        self.circle4_=  [[0, 4], [1, 4], [2, 3], [3, 2],
            [4, 1], [4, 0], [4, -1], [3, -2],
            [2, -3], [1, -4], [0, -4], [-1, -4],
            [-2, -3], [-3, -2], [-4, -1], [-4, 0],
            [-4, 1], [-3, 2], [-2, 3], [-1, 4]]
        
        self.sae_ = [np.zeros((self.sensor_height_, self.sensor_width_)), np.zeros((self.sensor_height_, self.sensor_width_))]


    def isFeature(self, e):
        # update SAE
        pol = 0
        if e.polarity:
            pol = 1


        self.sae_[pol][e.x, e.y] = e.t

        max_scale = 1

        # only check if not too close to border
        cs = max_scale*4
        if e.x < cs or e.x >= self.sensor_width_-cs or \
             e.y < cs or e.y >= self.sensor_height_-cs:     
            return False

        found_streak = False

        for i in range(16):
            for streak_size in range(3,6+1):
                
                # check that streak event is larger than neighbor
                if self.sae_[pol][e.x+self.circle3_[i][0], e.y+self.circle3_[i][1]] <  self.sae_[pol][e.x+self.circle3_[(i-1+16)%16][0], e.y+self.circle3_[(i-1+16)%16][1]]:
                    continue

                # check that streak event is larger than neighbor
                if self.sae_[pol][e.x+self.circle3_[(i+streak_size-1)%16][0], e.y+self.circle3_[(i+streak_size-1)%16][1]] < self.sae_[pol][e.x+self.circle3_[(i+streak_size)%16][0], e.y+self.circle3_[(i+streak_size)%16][1]]:
                    continue

                min_t = self.sae_[pol][e.x+self.circle3_[i][0], e.y+self.circle3_[i][1]]
                for j in range(1, streak_size):
                    tj = self.sae_[pol][e.x+self.circle3_[(i+j)%16][0], e.y+self.circle3_[(i+j)%16][1]]
                    if tj < min_t:
                        min_t = tj
                

                did_break = False
                for j in range(streak_size, 16):
                    tj = self.sae_[pol][e.x+self.circle3_[(i+j)%16][0], e.y+self.circle3_[(i+j)%16][1]]

                    if tj >= min_t:
                        did_break = True
                        break
                        
                if not did_break:
                    found_streak = True
                    break
                
                if found_streak:
                    break
                
        if found_streak:
        
            found_streak = False
            for i in range(20):
            
                for streak_size in range(4, 8+1):
                
                    # check that first event is larger than neighbor
                    if self.sae_[pol][e.x+self.circle4_[i][0], e.y+self.circle4_[i][1]] <  self.sae_[pol][e.x+self.circle4_[(i-1+20)%20][0], e.y+self.circle4_[(i-1+20)%20][1]]:
                        continue

                    # check that streak event is larger than neighbor
                    if self.sae_[pol][e.x+self.circle4_[(i+streak_size-1)%20][0], e.y+self.circle4_[(i+streak_size-1)%20][1]] < self.sae_[pol][e.x+self.circle4_[(i+streak_size)%20][0], e.y+self.circle4_[(i+streak_size)%20][1]]:
                        continue

                    min_t = self.sae_[pol][e.x+self.circle4_[i][0], e.y+self.circle4_[i][1]]
                    for j in range(1, streak_size):
                    
                        tj = self.sae_[pol][e.x+self.circle4_[(i+j)%20][0], e.y+self.circle4_[(i+j)%20][1]]
                        if tj < min_t:
                            min_t = tj
                        

                    did_break = False
                    for j in range(streak_size, 20):
                        tj = self.sae_[pol][e.x+self.circle4_[(i+j)%20][0], e.y+self.circle4_[(i+j)%20][1]]
                        if tj >= min_t:
                            did_break = True
                            break
                            

                    if not did_break:
                        found_streak = True
                        break
                    
                if found_streak:
                    break

        return found_streak


    def isFeature2(self, ee):
        x = ee[0]
        y = ee[1]
        t = ee[2]
        polarity = ee[3]

        # update SAE
        pol = 0
        if polarity:
            pol = 1


        self.sae_[pol][x, y] = t

        max_scale = 1

        # only check if not too close to border
        cs = max_scale*4
        if x < cs or x >= self.sensor_width_-cs or \
             y < cs or y >= self.sensor_height_-cs:     
            return False

        found_streak = False
        
        for i in range(16):
            for streak_size in range(3,6+1):
                
                # check that streak event is larger than neighbor
                # if self.sae_[pol][e.x+self.circle3_[i][0], e.y+self.circle3_[i][1]] <  self.sae_[pol][e.x+self.circle3_[(i-1+16)%16][0], e.y+self.circle3_[(i-1+16)%16][1]]:
                if self.get_sae_c3(x, y, pol, i) < self.get_sae_c3(x, y, pol, (i-1+16)%16):
                    continue

                # check that streak event is larger than neighbor
                # if self.sae_[pol][e.x+self.circle3_[(i+streak_size-1)%16][0], e.y+self.circle3_[(i+streak_size-1)%16][1]] < self.sae_[pol][e.x+self.circle3_[(i+streak_size)%16][0], e.y+self.circle3_[(i+streak_size)%16][1]]:
                if self.get_sae_c3(x, y, pol, (i+streak_size-1)%16) < self.get_sae_c3(x, y, pol, (i+streak_size)%16):
                    continue

                # min_t = self.sae_[pol][e.x+self.circle3_[i][0], e.y+self.circle3_[i][1]]
                min_t = self.get_sae_c3(x, y, pol, i)
                for j in range(1, streak_size):
                    # tj = self.sae_[pol][e.x+self.circle3_[(i+j)%16][0], e.y+self.circle3_[(i+j)%16][1]]
                    tj = self.get_sae_c3(x, y, pol, (i+j)%16)

                    if tj < min_t:
                        min_t = tj
                

                did_break = False
                for j in range(streak_size, 16):
                    # tj = self.sae_[pol][e.x+self.circle3_[(i+j)%16][0], e.y+self.circle3_[(i+j)%16][1]]
                    tj = self.get_sae_c3(x, y, pol, (i+j)%16)


                    if tj >= min_t:
                        did_break = True
                        break
                        
                if not did_break:
                    found_streak = True
                    break
                
                if found_streak:
                    break
                
        if found_streak:
        
            found_streak = False

            for i in range(20):
            
                for streak_size in range(4, 8+1):
                
                    # check that first event is larger than neighbor
                    # if self.sae_[pol][e.x+self.circle4_[i][0], e.y+self.circle4_[i][1]] <  self.sae_[pol][e.x+self.circle4_[(i-1+20)%20][0], e.y+self.circle4_[(i-1+20)%20][1]]:
                    if self.get_sae_c4(x, y, pol, i) < self.get_sae_c4(x, y, pol, (i-1+20)%20):
                        continue

                    # check that streak event is larger than neighbor
                    # if self.sae_[pol][e.x+self.circle4_[(i+streak_size-1)%20][0], e.y+self.circle4_[(i+streak_size-1)%20][1]] < self.sae_[pol][e.x+self.circle4_[(i+streak_size)%20][0], e.y+self.circle4_[(i+streak_size)%20][1]]:
                    if self.get_sae_c4(x, y, pol, (i+streak_size-1)%20) < self.get_sae_c4(x, y, pol, (i+streak_size)%20):
                        continue

                    # min_t = self.sae_[pol][e.x+self.circle4_[i][0], e.y+self.circle4_[i][1]]
                    min_t = self.get_sae_c4(x, y, pol, i)

                    for j in range(1, streak_size):
                    
                        # tj = self.sae_[pol][e.x+self.circle4_[(i+j)%20][0], e.y+self.circle4_[(i+j)%20][1]]
                        tj = self.get_sae_c4(x, y, pol, (i+j)%20)

                        if tj < min_t:
                            min_t = tj
                        

                    did_break = False
                    for j in range(streak_size, 20):
                        # tj = self.sae_[pol][e.x+self.circle4_[(i+j)%20][0], e.y+self.circle4_[(i+j)%20][1]]
                        tj = self.get_sae_c4(x, y, pol, (i+j)%20)

                        if tj >= min_t:
                            did_break = True
                            break
                            

                    if not did_break:
                        found_streak = True
                        break
                    
                if found_streak:
                    break

        return found_streak
    
    def get_sae_c3(self, x, y, pol, i):
        return self.sae_[pol][x+self.circle3_[i][0], y+self.circle3_[i][1]] 

    def get_sae_c4(self, x, y, pol, i):
        return self.sae_[pol][x+self.circle4_[i][0], y+self.circle4_[i][1]] 
