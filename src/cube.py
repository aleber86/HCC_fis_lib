import numpy as np
from cilinder import Cilinder


class Cube(Cilinder):
    """Primitive mesh body. Child of Cilinder class."""
    def __init__(self, side_size, mass, friction,
                 init_pos, init_angular, init_vel,
                 init_rot, destructive, _wp = np.float64):

        Cilinder.__init__(self, side_size, side_size,4 ,mass,
                          friction, init_pos, init_angular,
                          init_vel, init_rot, destructive, _wp = _wp )
    def __inertia_tensor(self):
        diag_1_1 = 1/6*self.mass*(self.size**2)
        diag_2_2 = diag_1_1
        diag_3_3 = diag_1_1
        self.inertia_tensor = np.array([[diag_1_1,0,0],[0,diag_2_2,0],[0,0,diag_3_3]])

class Prisma(Cilinder):
    """Primitive mesh body. Child of Cilinder class."""
    def __init__(self, side_size, height, mass, friction,
                 init_pos, init_angular,init_vel, init_rot,
                 destructive, _wp = np.float64):

        Cilinder.__init__(self, side_size, height, 3 ,mass,
                          friction, init_pos, init_angular, init_vel,
                          init_rot, destructive, _wp = _wp )
