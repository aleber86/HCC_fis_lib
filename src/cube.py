import numpy as np
from cilinder import Cilinder


class Cube(Cilinder):
    """Primitive mesh body. Child of Cilinder class."""
    def __init__(self, side_size, mass, friction, init_pos, init_vel, init_rot, destructive, _wp = np.float64):
        Cilinder.__init__(self, side_size, side_size,4 ,mass, friction, init_pos, init_vel, init_rot, destructive, _wp = _wp )


class Prisma(Cilinder):
    """Primitive mesh body. Child of Cilinder class."""
    def __init__(self, side_size, height, mass, friction, init_pos, init_vel, init_rot, destructive, _wp = np.float64):
        Cilinder.__init__(self, side_size, height, 3 ,mass, friction, init_pos, init_vel, init_rot, destructive, _wp = _wp )
