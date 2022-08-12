

class A:
    '''
    An abstract base class for low level controllers.
    '''
    ## def __init__(self, sim):
    ##     self.sim = sim
    ##     self.action_space = None

    def __init__(self, a):
        print(a)

class B(A):
    '''
    An abstract base class for low level controllers.
    '''
    ## def __init__(self, sim):
    ##     self.sim = sim
    ##     self.action_space = None

    def __init__(self, a, b):
        super(B, self).__init__(a)
        print("#", b)

class C(B):
    '''
    An abstract base class for low level controllers.
    '''
    ## def __init__(self, sim):
    ##     self.sim = sim
    ##     self.action_space = None

    def __init__(self, a, b, c):
        super(C, self).__init__(a, b)
        print("##", c)

c = C(1, 2, 3)
