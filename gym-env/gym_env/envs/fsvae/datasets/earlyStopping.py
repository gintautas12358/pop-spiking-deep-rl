
class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.old_loss = 1e5

    # def __call__(self, train_loss, validation_loss):
    #     if (validation_loss - train_loss) > self.min_delta:
    #         self.counter +=1
    #         if self.counter >= self.tolerance:  
    #             self.early_stop = True

    def __call__(self, test_loss):
        if (test_loss - self.old_loss ) > self.min_delta:
            self.counter +=1
            print("counter", self.counter)
            if self.counter >= self.tolerance:  
                self.early_stop = True

        if test_loss < self.old_loss:
            self.old_loss = test_loss
            self.counter = 0 