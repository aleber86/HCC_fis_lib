import numpy as np


class Particle:
    def __init__(self, mass : float, init_pos : list or np.array = [0., 0., 0.],
                 init_vel : list or np.array = [0., 0., 0.],
                 init_external_force : list or np.array = [0.,0.,0.] , friction = 0.8, _wp = np.float64):
        self._wp = _wp
        self.position = np.array(init_pos, dtype = _wp)
        self.velocity = np.array(init_vel, dtype= _wp)
        self.external_force_interaction = np.array(init_external_force, dtype = _wp)
        self.mass = _wp(mass)
        self.momentum = self.mass * self.velocity
        self.remove = False #Existence: {True : subscribe, False : unsuscribe}
        self.friction = friction
        self.hit_box_global = None
        self.__collision_box()



    def get_external_force(self):
        return self.external_force_interaction

    def get_mass(self):
        return self.mass

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity

    def get_state(self):
        return self.remove

    def get_friction(self):
        return self.friction

    def get_momentum(self):
        return self.momentum

    def set_external_force(self, external):
        self.external_force_interaction = external

    def set_position(self, actual_position : list or np.array):
        self.position = actual_position

    def set_velocity(self, actual_velocity : list or np.array):
        self.velocity = actual_velocity

    def set_state(self, actual_state : bool):
        self.remove = actual_state

    def set_momentum(self):
        self.momentum = self.mass * self.velocity

    def collision_box(self):
        return self.hit_box_global

    def __collision_box(self):
        origin = self.get_position()
        max_x, max_y, max_z = origin
        min_x, min_y, min_z = origin
        offset =  1.e-1
        max_x += offset
        max_y += offset
        max_z += offset
        min_x -= offset
        min_y -= offset
        min_z -= offset
        self.hit_box_global = np.array([[min_x, min_y, min_z],
                                       [min_x, max_y, min_z],
                                       [min_x, max_y, max_z],
                                       [max_x, min_y, min_z],
                                       [max_x, max_y, min_z],
                                       [max_x, max_y, max_z],
                                       [min_x, min_y, max_z],
                                       [max_x, min_y, max_z]
                                            ], dtype = self._wp)

    def kinetic_energy(self):
        velocity = self.get_velocity()
        k_energy = self._wp(0.5) *self.mass * np.dot(velocity, velocity)
        return k_energy


    def update(self, time=0.1):
        self.set_momentum()
        self.__collision_box()


if __name__ == '__main__':
    part = Particle(0,[0,0,0], [1,1,1])
    for _ in range(100):
        print(part.collision_box())
        part.update()
