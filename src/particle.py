import numpy as np


class Particle:
    def __init__(self,size_radius : float,  mass : float, init_pos : list or np.array = [0., 0., 0.],
                 init_vel : list or np.array = [0., 0., 0.],
                 init_external_force : list or np.array = [0.,0.,0.] , friction = 0.8, _wp = np.float64):
        self._wp = _wp
        self.size_radius = _wp(size_radius)
        self.position = np.array(init_pos, dtype = _wp)
        self.velocity = np.array(init_vel, dtype= _wp)
        self.external_force_interaction = np.array(init_external_force, dtype = _wp)
        self.mass = _wp(mass)
        self.momentum = self.mass * self.velocity
        self.remove = False #Existence: {True : subscribe, False : unsuscribe}
        self.friction = friction
        self.hit_box_global = None
        self.angular_rotation = np.array(np.zeros(3,), dtype=_wp)
        self.__collision_box()
        I_xx = 2./5.*self.mass*self.size_radius**2
        self.inertia_tensor = np.matrix([[I_xx,0,0],[0,I_xx,0],[0,0,I_xx]])
        self.inertia_tensor_inverse = np.linalg.inv(self.inertia_tensor)
        self.inertia_tensor_original = self.inertia_tensor
        self.rotational_velocity = np.array([0,0,0])

    def get_inertia_tensor_inverse(self) -> np.array:
        """Getter method. Gets inverse inertia tensor

        Args:
            None

        Returns:
            np.ndarray

        """
        return self.inertia_tensor_inverse

    def get_inertia_tensor(self) -> np.array:
        """Getter method. Gets inertia tensor

        Args:
            None

        Returns:
            np.ndarray

        """
        return self.inertia_tensor

    def set_rotation_velocity(self, rotation) -> None:
        """Setter method. Sets new velocity of rotation

        Args:
            rotation : np.array; new angular rotation (angular velocity).

        Returns:
            None
        """
        self.rotational_velocity = rotation

    def get_rotation_velocity(self) -> np.array:
        """Getter method. Gets angular velocity of body.

        Args:
            None

        Returns:
            np.array
        """
        return self.rotational_velocity

    def get_external_force(self) -> np.array:
        return self.external_force_interaction

    def get_mass(self) -> float:
        """Getter method. Gets inertial mass of body

        Args:
            None

        Returns:
            float

        """
        return self.mass

    def get_position(self) -> np.array:
        """Getter method. Gets center position of body.

        Args:
            None

        Returns:
            np.array
        """
        return self.position

    def get_velocity(self) -> np.array:
        """Getter method. Gets linear velocity of body center.

        Args:
            None

        Returns:
            np.array
        """
        return self.velocity

    def get_state(self) -> bool:
        """Getter method. Truth state of body. Attribute to be
        used by other classes.

        Args:
            None
        Returns:
            bool
        """
        return self.remove

    def get_friction(self) -> float:
        """Getter method. Gets value of friction.

        Args:
            None

        Returns:
            float
        """
        return self.friction

    def get_momentum(self) -> np.array:
        return self.momentum

    def get_size(self) -> float:
        return self.size_radius

    def set_external_force(self, external) -> None:
        self.external_force_interaction = external

    def set_position(self, actual_position : list or np.array) -> None:
        """Setter method. Sets new position of center.

        Args:
            position : np.array; new center position of body.

        Returns:
            None
        """
        self.position = actual_position

    def set_velocity(self, actual_velocity : list or np.array) -> None:
        """Setter method. Sets new linear velocity of center.

        Args:
            actual_velocity : np.array; new linear velocity of center.

        Returns:
            None
        """
        self.velocity = actual_velocity

    def set_state(self, actual_state : bool) -> None:
        """Setter method. Truth state of body. Attribute to be
        used by other classes.

        Args:
            state : bool

        Returns:
            None
        """
        self.remove = actual_state

    def set_momentum(self) -> None:
        self.momentum = self.mass * self.velocity

    def collision_box(self) -> np.ndarray:
        """Returns hit box in global coordinates.

        Args:
            None

        Returns:
            np.ndarray
        """
        return self.hit_box_global

    def __collision_box(self) -> None:
        """Method calculates collision box (local coordinates).

        Args:
            None

        Returns:
            None

        """
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

    def kinetic_energy(self) -> float:
        """Method calculates kinetic energy

        Args:
            None

        Returns:
            float : energy
        """
        velocity = self.get_velocity()
        k_energy = self._wp(0.5) *self.mass * np.dot(velocity,velocity)
        rot = 0.5*np.dot(self.rotational_velocity, self.angular_momentum())
        return k_energy + rot

    def angular_momentum(self) -> np.array:
        """Calculates angular momentum (global system of coordinates).

        Args:
            None

        Returns:
            np.array
        """
        if np.linalg.norm(self.rotational_velocity)>1.e-15:

            rotational_velocity = np.reshape(self.rotational_velocity, (3,1))
            ang_rot = np.matmul(self.inertia_tensor_inverse, rotational_velocity)
            ang_rot = np.reshape(np.array(ang_rot), (3,))
        else:
            ang_rot = np.cross(self.position, self.mass*self.velocity)
        return  ang_rot

    def update(self) -> None:
        """Updates every element inside the class."""
        self.__collision_box()

    def linear_momentum(self) -> np.array:
        """Calculates linear momentum (global system of coordinates).

        Args:
            None

        Returns:
            np.array
        """
        lin_mom = self.mass * self.velocity
        return lin_mom

if __name__ == '__main__':
    part = Particle(0,[0,0,0], [1,1,1])
    for _ in range(100):
        print(part.collision_box())
        part.update()
