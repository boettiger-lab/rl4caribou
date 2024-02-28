import numpy as np

class constAction:
    def __init__(self, mortality_vec=np.zeros(2, dtype=np.float32), env = None, **kwargs):
        #
        # preprocess
        if isinstance(mortality_vec, list):
            mortality_vec = np.float32(mortality_vec)
        #
        self.mortality_vec = mortality_vec
        self.action = 2 * self.mortality_vec - 1
        self.env = env

    def predict(self, observation, **kwargs):
        return self.action, {}