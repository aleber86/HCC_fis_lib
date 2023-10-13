import numpy as np


class Particle:
    def __init__(self, mass : float, init_pos : list or np.array = [0., 0., 0.],
                 init_vel : list or np.array = [0., 0., 0.],
                 init_external_force : list or np.array = [0.,0.,0.] , friction = 0.8, _wp = np.float64):

        self.position = np.array(init_pos, dtype = _wp)
        self.velocity = np.array(init_vel, dtype= _wp)
        self.external_force_interaction = np.array(init_external_force, dtype = _wp)
        self.mass = _wp(mass)
        self.remove = False #Existence: {True : subscribe, False : unsuscribe}
        self.friction = friction

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

    def set_external_force(self, external):
        self.external_force_interaction = external

    def set_position(self, actual_position : list or np.array):
        self.position = actual_position

    def set_velocity(self, actual_velocity : list or np.array):
        self.velocity = actual_velocity

    def set_state(self, actual_state : bool):
        self.remove = actual_state

