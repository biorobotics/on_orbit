import numpy as np 

class PlanarImpedance(object):

    def __init__(self):
      n = 0
      self.Bp = np.diag(np.array([n*0,n*0,n*0]))
      self.Kp = np.diag(np.array([n*50,n*50,n*0]))

    def compute_force(self, nozzle_pos, nozzle_vel, ref_nozzle_pos, ref_nozzle_vel):
      return self.Bp@(nozzle_vel - ref_nozzle_vel) + self.Kp@(nozzle_pos - ref_nozzle_pos) 
      