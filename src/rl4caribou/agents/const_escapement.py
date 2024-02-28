import numpy as np

class constEsc:
    def __init__(self, escapement_vec, env = None):
        #
        # preprocess
        if isinstance(escapement_vec, list):
            escapement_vec = np.float32(escapement_vec)
        escapement_vec = np.clip(
            escapement_vec, a_min = 0, a_max = None
        )
        #
        self.escapement_vec = escapement_vec
        self.env = env
        self.bound = 1
        if self.env is not None:
            self.bound = self.env.bound

    def predict(self, observation, **kwargs):
        obs_nat_units = self.bound * self.to_01(observation)
        m_mort = self.moose_mortality(obs_nat_units[0])
        w_mort = self.wolf_mortality(obs_nat_units[3])
        mortality = np.float32([m_mort, w_mort])
        return self.to_pm1(mortality), {}

    def moose_mortality(self, moose_pop):
        if moose_pop <= self.escapement_vec[0]:
            return 0
        else:
            return (moose_pop - self.escapement_vec[0]) / moose_pop
    
    def wolf_mortality(self, wolf_pop):
        if wolf_pop <= self.escapement_vec[1]:
            return 0
        else:
            return (wolf_pop - self.escapement_vec[1]) / wolf_pop
    
    def to_01(self, val):
        return (val + 1 ) / 2

    def to_pm1(self, val):
        return 2 * val - 1


        
