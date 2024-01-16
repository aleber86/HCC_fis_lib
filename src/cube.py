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

if __name__ == '__main__':
    from object_class_module import render
    size = 1
    cube = Cube(size,1,0,[1,1,1],[0,0,0],[0,0,0],[0,0,0], False)
    objects = [cube]
    cube_volume_numeric = cube.volume_calc()
    render(objects, 1)
    cube_volume_analitic = size**3
    relative_error_v = np.abs(cube_volume_numeric - cube_volume_analitic)/np.abs(cube_volume_analitic)*100
    print(f"Cube numeric volume = {cube_volume_numeric}; Cube analitic volume = {cube_volume_analitic}")
    print(f"Relative error [%] = {relative_error_v}")
    print()
    cube_surface_numeric = cube.total_surface_calc()
    cube_surface_analitic = 6*size**2
    relative_error_s = np.abs(cube_surface_numeric - cube_surface_analitic)/np.abs(cube_surface_analitic)*100
    print(f"Cube numeric surface = {cube_surface_numeric}; Cube analitic volume = {cube_surface_analitic}")
    print(f"Relative error [%] = {relative_error_s}")
