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
        self.__inertia_tensor(side_size)
    def __inertia_tensor(self, side_size):
        side_size = side_size
        diag_1_1 = 1/6*self.mass*(side_size**2)
        diag_2_2 = diag_1_1
        diag_3_3 = diag_2_2
        self.inertia_tensor = np.array([[diag_1_1,0,0],[0,diag_2_2,0],[0,0,diag_3_3]])
        self.inertia_tensor_inverse = np.linalg.inv(self.inertia_tensor)

class Prisma(Cilinder):
    """Primitive mesh body. Child of Cilinder class."""
    def __init__(self, side_size, height, mass, friction,
                 init_pos, init_angular,init_vel, init_rot,
                 destructive, _wp = np.float64):

        Cilinder.__init__(self, side_size, height, 3 ,mass,
                          friction, init_pos, init_angular, init_vel,
                          init_rot, destructive, _wp = _wp )
