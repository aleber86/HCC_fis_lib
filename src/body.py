import numpy as np

class Body:
    r"""Class of 3D objects (mesh only).
    mass : represents the mass of the body
    vertex : sets the position of every vertex that defines the bodys
    on local coordinates.
    [[x_1,y_1,z_1],...,[x_n, y_n, z_n]]
    friction : represnts bodys friction
    init_pos/init_vel/init_rot : initial phase space
    init_pos : [x,y,z,\phi,\theta,\Phi] (body center)
    init_vel : [\dot{x}, \dot{y}, \dot{z}]
    init_rot : [\dot{\phi}, \dot{\theta}, \dot{\Phi}]
    destructive : if True, it breaks on collision
    edges : defines 2d tuples for edges of the 3D object
    [[v_1, v_2], [v_1, v_3], ...., [v_i, v_k]]
    _wp : working precision
    """
    _tolerance = 1.e-14
    def __init__(self, mass : float, vertex, friction : float,
                  init_pos : list or np.array, init_angular: list or np.array,
                 init_vel : list or np.array,
                  init_rot : list or np.array, destructive : bool,
                  edges_indexes, _wp = np.float64):
        self.offset_rotation_axis = None
        self._wp = _wp
        self.mass = _wp(mass)
        self.friction = _wp(friction)
        self.position = np.array(init_pos, dtype = _wp)
        self.angular_position = np.array(init_angular, dtype=_wp)
        self.linear_velocity = np.array(init_vel, dtype = _wp)
        self.rotation_velocity = np.array(init_rot, dtype = _wp)
        self.destructive = destructive
        self.edges_indexes = np.array(edges_indexes, dtype = np.int32)
        self.hit_box_local = np.array([0,0,0])
        self.hit_box_global = np.array([0,0,0])
        self.local_vertex = _wp(vertex)
        self.global_vertex = 0.0
        self.array_point = np.array([[0,0,0]])
        self.inertia_tensor = np.zeros((3,3), dtype = _wp)
        self.inertia_tensor_inverse = np.zeros((3,3), dtype = _wp)
        self.inertia_tensor_original = np.zeros((3,3), dtype = _wp)
        self.inertia_tensor_inverse_original = np.zeros((3,3), dtype = _wp)
        self.__collision_box()
        self.__vertex_position()
        self.edges = self.edges_change()
        self.axial_vectors_to_faces = None
        self.state = True
        self.surface_vector = None
        self.change_variable_state_position = True
        self.change_variable_state_angular = True
        self.change_variable_state_rotation_v = True
        self.change_variable_state_vel = True
        self._tolerance = _wp(self._tolerance)
        self.update_faces()
        self.volume = 0.0 #Re-defined in volume_calc()
        self.area = 0.0 #Re-defined in total_surface_calc
        self.collision_response = False
        self.last_vertex_rotation = self.local_vertex
        self.last_hit_box_rotation = self.hit_box_local


    def update_faces(self) -> None:
        """Method updates every vertex on the faces
            Agrs : None
            Returns: None

        """
        if self.faces is not None:
            offset = 0
            for index,face in enumerate(self.faces):
                vertex_len = len(face.get_vertex_position())
                self.faces[index].set_vertex_position(self.global_vertex[offset:offset+vertex_len])
                offset += vertex_len
                self.faces[index].vector_surface()


    def change_status(self, variable_change, variable_to_change) -> bool:
        """It determines if phase state variables change. If not
        there will be no rotations needed. Reduce computational
        usage.

        Args:
            variable_change : last value to check.
            variable_to_change : actual value to check

        Returns:
            bool

        """

        state = np.sum(np.abs(variable_change - variable_to_change))
        if state > self._tolerance:
            flag = True
        else:
            flag = False
        return flag

    def volume_calc(self, samples_quant = 2000, n = 10) -> float:
        """Method implemented by Monte Carlo method
        to calculate volume of 3D bodys

        Args:
            samples_quant : int; quantity of random points to check.
            n : int; number of samples.

        Returns:
            float : volume
        """
        volume_array = []
        for _ in np.arange(n):
            counter = [0]
            max_x, min_x = np.max(self.global_vertex[:,0]), np.min(self.global_vertex[:,0])
            max_y, min_y = np.max(self.global_vertex[:,1]), np.min(self.global_vertex[:,1])
            max_z, min_z = np.max(self.global_vertex[:,2]), np.min(self.global_vertex[:,2])
            max_vector = np.array([max_x, max_y, max_z])
            min_vector = np.array([min_x, min_y, min_z])
            surface_encaps = np.prod([np.abs(max_vector[i]-min_vector[i]) for i in np.arange(3)])
            array_points_random = np.array([[np.random.uniform(min_vector[0], max_vector[0]),
                                             np.random.uniform(min_vector[1], max_vector[1]),
                                             np.random.uniform(min_vector[2], max_vector[2])]
                                            for _ in np.arange(samples_quant)])
            self.array_point = array_points_random
            np.apply_along_axis(self._point_calculation, 1, array_points_random, counter)
            monte_carlo_area = surface_encaps * counter[0]/samples_quant
            volume_array.append(monte_carlo_area)

        self.volume = np.mean(volume_array)

        return self.volume

    def _point_calculation(self, point, counter) -> None:
        """Method checks if a point is inside a body. It
        uses normal versors of faces.

        Args:
            point : np.array; point to check.
            counter : one element list; accumulator of points inside

        Returns:
            None
        """


        bool_array = []
        for face in self.faces:
            position = face.get_position()
            normal_versor = face.get_surface_vectors()
            if np.dot(point-position, normal_versor-position)<=0.:
                bool_array.append(True)
            else:
                bool_array.append(False)
        if np.sum(bool_array)==len(self.faces):
            counter[0] = counter[0]+1

    def total_surface_calc(self, quant = 1000, n=10) -> float:
        """Method calculates surface of a body using a Monte Carlo method.
        Takes the surfaces of faces.

        Args:
            quant : int; number or random points to check.
            n : int; number of samples.

        Returns:
            float : total surface
        """
        if self.faces is not None:
            total_surface_per_face = np.array(
                [face.surface_calc(quant, n) for face in self.faces], dtype=self._wp)
            total_surface = np.sum(total_surface_per_face)
            self.area = total_surface
            return total_surface


    def density_calc(self):
        """Method sets the density of bodys. All bodys are treated as homogenous.
        rho = mass / volume

        Args:
            None

        Returns:
            float
        """
        dens = self.mass/self.volume
        return dens

    def get_collision_response(self) -> bool:
        """Getter method. Gets collision response

        Args:
            None

        Returns:
            bool
        """
        return self.collision_response

    def get_mass(self) -> float:
        """Getter method. Gets inertial mass of body

        Args:
            None

        Returns:
            float

        """
        return self.mass

    def get_edges(self) -> np.ndarray:
        """Getter method. Gets edges [base, point]

        Args:
            None

        Returns:
            np.ndarray

        """
        return self.edges

    def get_inertia_tensor(self) -> np.ndarray:
        """Getter method. Gets inertia tensor

        Args:
            None

        Returns:
            np.ndarray

        """
        return self.inertia_tensor

    def get_inertia_tensor_inverse(self) -> np.ndarray:
        """Getter method. Gets inverse inertia tensor

        Args:
            None

        Returns:
            np.ndarray

        """
        return self.inertia_tensor_inverse

    def get_velocity(self) -> np.array:
        """Getter method. Gets linear velocity of body center.

        Args:
            None

        Returns:
            np.array
        """
        return self.linear_velocity

    def get_position(self) -> np.array:
        """Getter method. Gets center position of body.

        Args:
            None

        Returns:
            np.array
        """
        return self.position

    def get_angular_position(self) -> np.array:
        """Getter method. Gets angular position of body.

        Args:
            None

        Returns:
            np.array
        """
        return self.angular_position

    def get_rotation_velocity(self) -> np.array:
        """Getter method. Gets angular velocity of body.

        Args:
            None

        Returns:
            np.array
        """
        return self.rotation_velocity

    def get_vertex_position(self) -> np.ndarray:
        """Getter method. Gets edge vertex position in global coordinates
        of body as array.

        Args:
            None

        Returns:
            np.ndarray
        """
        return self.global_vertex

    def get_vertex_position_local(self) -> np.ndarray:
        """Getter method. Gets edge vertex position in local coordinates
        of body as array.

        Args:
            None

        Returns:
            np.ndarray

        """
        return self.local_vertex

    def get_edges_index(self) -> np.ndarray:
        """Getter method. Gets edge indexes of body as array.

        Args:
            None

        Returns:
            np.ndarray

        """
        return self.edges_indexes

    def get_faces(self) -> np.ndarray:
        """Getter method. Gets faces of body as array.

        Args:
            None

        Returns:
            np.ndarray

        """
        return self.faces

    def get_friction(self) -> float:
        """Getter method. Gets value of friction.

        Args:
            None

        Returns:
            float
        """
        return self.friction

    def get_state(self) -> bool:
        """Getter method. Truth state of body. Attribute to be
        used by other classes.

        Args:
            None
        Returns:
            bool
        """
        return self.state

    def get_offset_rotation_axis(self) -> np.array:
        """Getter method. Gets offset axis for rotations of the body.

        Args:
            None

        Returns:
            np.array; angular offset axis.
        """
        return self.offset_rotation_axis

    def set_collision_response(self, state : bool) -> None:
        """Setter method. Truth state of body. Attribute to be
        used by other classes.

        Args:
            state : bool

        Returns:
            None
        """
        self.collision_response = state

    def set_state(self, state : bool) -> None:
        """Setter method. Truth state of body. Attribute to be
        used by other classes.

        Args:
            state : bool

        Returns:
            None
        """
        self.state = state

    def set_offset_rotation_axis(self, axis : np.array) -> None:
        """Setter method. Sets offset axis for rotations of the body.

        Args:
            axis : np.array; new angular offset axis.

        Returns:
            None
        """
        self.offset_rotation_axis = axis

    def set_axial_vector_to_faces(self, vectors_to_center : np.array) -> None:
        """Setter method. Sets offset axis for rotations of faces.

        Args:
            vectors_to_center : np.array; new angular offset axis.

        Returns:
            None
        """
        self.axial_vectors_to_faces = vectors_to_center

    def set_angular_position(self, angular_position : np.array) ->None:
        """Setter method. Sets new angular position of the body.

        Args:
            angular_position : np.array; new angular orientation of body.

        Returns:
            None
        """
        self.change_variable_state_angular = self.change_status(angular_position,
                                                                self.angular_position)
        if self.change_variable_state_angular:
            self.angular_position = np.fmod(angular_position, 2.*np.pi)

    def set_edges(self, edg : np.ndarray) -> None:
        """Setter method. Sets new edges array.

        Args:
            edg : np.ndarray; edges [base, point] in global coordinates.

        Returns:
            None

        """
        self.edges = edg

    def set_inertia_tensor(self, inertia_tensor) -> None:
        """Setter method. Sets inertia tensor

        Args:
            None

        Returns:
            None

        """
        self.inertia_tensor = inertia_tensor

    def set_inertia_tensor_inverse(self) -> None:
        """Setter method. Sets inverse inertia tensor

        Args:
            None

        Returns:
            None

        """
        self.inertia_tensor_inverse = np.linalg.inv(self.inertia_tensor)

    def set_velocity(self, velocity : np.array) -> None:
        """Setter method. Sets new linear velocity of center.

        Args:
            velocity : np.array; new linear velocity of center.

        Returns:
            None
        """
        self.change_variable_state_vel = self.change_status(velocity,
                                                            self.linear_velocity)
        if self.change_variable_state_vel:
            self.linear_velocity = velocity

    def set_position(self, position : np.array) -> None:
        """Setter method. Sets new position of center.

        Args:
            position : np.array; new center position of body.

        Returns:
            None
        """


        self.change_variable_state_position = self.change_status(position,
                                                                 self.position)
        if self.change_variable_state_position:
            self.position = position


    def set_rotation_velocity(self, rotation) -> None:
        """Setter method. Sets new velocity of rotation

        Args:
            rotation : np.array; new angular rotation (angular velocity).

        Returns:
            None
        """


        self.change_variable_state_rotation_v = self.change_status(rotation,
                                                                   self.rotation_velocity)
        if self.change_variable_state_rotation_v:
            self.rotation_velocity = rotation


    def set_vertex_position(self, position_of_vertex : np.array) -> None:
        """Setter method. Sets position of vertex in global coordinates

        Args:
            position_of_vertex : new position to set.

        Returns:
            None
        """


        self.global_vertex = position_of_vertex

    def rotation_matrix(self, angular_deviation) -> np.ndarray:
        """Method defines the rotation matrix

        Args:
            angular_deviation: np.array; angular position of body.

        Returns:
            rotation_matrix: np.array; new matrix of rotation.
        """

        psi, theta, phi = angular_deviation
        row_1 = [np.cos(theta)*np.cos(phi), np.sin(psi)*np.sin(theta)*np.cos(phi)-\
                 np.cos(psi)*np.sin(phi), np.cos(psi)*np.sin(theta)*np.cos(phi)+\
                 np.sin(psi)*np.sin(phi)]
        row_2 = [np.cos(theta)*np.sin(phi), np.sin(psi)*np.sin(theta)*np.sin(phi)+\
                 np.cos(psi)*np.cos(phi), np.cos(psi)*np.sin(theta)*np.sin(phi)-\
                 np.sin(psi)*np.cos(phi)]
        row_3 = [-np.sin(theta), np.sin(psi)*np.cos(theta), np.cos(psi)*np.cos(theta)]


        rotational_matrix = np.matrix([row_1, row_2, row_3], dtype=self._wp)
        return rotational_matrix

    def _vector_rotation(self, vector_to_rotate,
                         angular_deviation, origin) -> np.array:
        """Rotation matrix = R_z(alpha)*R_y(beta)*R_x(gamma) | RotMat * VECTOR

        Args:
            vector_to_rotate : np.array; vector pre-rotated.
            angular_deviation : np.array; position of faces (angular).
            origin : np.array; axis offset.

        Returns:
            np.array : vector rotated
        """


        rotational_matrix = self.rotation_matrix(angular_deviation)
        diff = vector_to_rotate - origin
        row = diff.shape[0]
        diff_column = np.reshape(diff, (row, 1))
        vector_rotated = np.matmul(rotational_matrix, diff_column)

        return vector_rotated

    def __vertex_position(self) -> None:
        """Updates center position and vertex position on global system
            If axis = None, the body rotates over an axis thus it center (not the mass center,
            body defined center by instant position)

        Args:
            None

        Returns:
            None
        """


        zero_vector = np.zeros((3,), dtype = self._wp)
        center_update_position = self.position
        if self.offset_rotation_axis is None:
            origin = zero_vector
        else:
            origin = self.offset_rotation_axis

        angular_update_direction = self.get_angular_position()
#        if True:
        if self.change_variable_state_angular:

            vertex_update_position = np.apply_along_axis(self._vector_rotation, 1, self.local_vertex,
                                                         angular_update_direction, origin )
            vertex_update_position = np.array(vertex_update_position)
            hit_box_update_position = np.apply_along_axis(self._vector_rotation, 1, self.hit_box_local,
                                                         angular_update_direction, origin )

            hit_box_update_position = np.array(hit_box_update_position)
            self.last_vertex_rotation = vertex_update_position
            self.last_hit_box_rotation = hit_box_update_position
        #else:
        #    vertex_update_position = self.last_vertex_rotation
        #    hit_box_update_position = self.hit_box_local + center_update_position
        self.hit_box_global = self.last_hit_box_rotation + center_update_position
        self.set_vertex_position(self.last_vertex_rotation + center_update_position)
        if self.faces is not None:
            positions_local= self.faces_local_position
            if True:
                R = self.rotation_matrix(angular_update_direction)
                R_T = np.transpose(R)
                self.inertia_tensor = np.matmul(np.matmul(R,self.inertia_tensor_original),R_T)
                self.inertia_tensor_inverse = np.matmul(np.matmul(R,self.inertia_tensor_inverse_original),R_T)

                position_new_local = np.apply_along_axis(self._vector_rotation, 1, positions_local,
                                                         angular_update_direction, origin)

                position_new_global = position_new_local
            else:
                position_new_global = positions_local
            for vec,face in zip(position_new_global,self.faces):
                vec1 = np.reshape(np.array(vec), (3,))
                face.set_position(vec1+center_update_position)

    def __define_edges(self, indexes) -> np.ndarray:
        """Method actualy change de edge using indexes

        Args:
            np.ndarray: int

        Returns:
            np.ndarray
        """
        edge = np.array([self.global_vertex[indexes[0]], self.global_vertex[indexes[1]]], dtype = self._wp)
        return edge

    def edges_change(self) -> np.ndarray:
        """Method changes every edge and returns the new iteration.

        Args:
            None

        Returns:
            np.ndarray
        """

        edges = np.apply_along_axis(self.__define_edges, 1, self.edges_indexes)
        return  edges

    def get_surface_vectors(self) -> np.array:
        """Getter method. Returns surface versors. Planes only one.
        Bodies returns each surface versor of every face.get_surface_vectors

        Args:
            None

        Returns:
            np.ndarray
        """
        return self.surface_vector

    def vector_surface(self) -> None:
        """Method of Body not defined. Every child class must overload
        the operator. Only Plane class has this method defined"""
        pass

    def _direction(self, face_vector, vector_to_proyect) -> bool:
        """Method checks if a collision could happen.
        A · B = |A||B| Cos(b). If A · B >= 0, then collision could occurre

        Args:
            face_vector : np.array; normal face versor.
            vector_to_proyect : np.array; vector to check.

        Returns:
            bool
        """

        condition = False
        if np.dot(face_vector, vector_to_proyect)>self._tolerance:
            condition = True
        return condition

    def collision_box(self) -> np.ndarray:
        """Returns hit box in global coordinates.

        Args:
            None

        Returns:
            np.ndarray
        """

        return self.hit_box_global

    def __collision_box(self):
        """Method calculates collision box (local coordinates).

        Args:
            None

        Returns:
            None

        """
        max_x = np.max(self.local_vertex[:,0])
        max_y = np.max(self.local_vertex[:,1])
        max_z = np.max(self.local_vertex[:,2])
        offset = (np.abs(max_x) + np.abs(max_y) + np.abs(max_z))/3 * 1
        max_x += offset
        max_y += offset
        max_z += offset
        min_x = np.min(self.local_vertex[:,0]) - offset
        min_y = np.min(self.local_vertex[:,1]) - offset
        min_z = np.min(self.local_vertex[:,2]) - offset
        self.hit_box_local = np.array([[min_x, min_y, min_z],
                                       [min_x, max_y, min_z],
                                       [min_x, max_y, max_z],
                                       [max_x, min_y, min_z],
                                       [max_x, max_y, min_z],
                                       [max_x, max_y, max_z],
                                       [min_x, min_y, max_z],
                                       [max_x, min_y, max_z]
                                       ], dtype = self._wp)

    def rotational_energy(self) -> float:
        """Method calculates only linear velocity kinetic energy

        Args:
            None

        Returns:
            float
        """
        rotational_velocity = self.rotation_velocity
        angular_momentum = np.reshape(np.array(self.angular_momentum()),(3,))
        rotational_velocity = np.reshape(np.array(rotational_velocity), (3,))
        rot_energy = 0.5 * np.dot(rotational_velocity, angular_momentum)
        return rot_energy

    def linear_energy(self) -> float:
        """Method calculates only linear velocity kinetic energy

        Args:
            None

        Returns:
            float
        """
        linear_velocity = self.get_velocity()
        mass = self.mass
        lin_energy = 0.5 *mass* np.dot(linear_velocity, linear_velocity)
        return lin_energy

    def angular_momentum(self) -> np.array:
        """Calculates angular momentum (global system of coordinates).

        Args:
            None

        Returns:
            np.array
        """
        rotational_velocity = self.rotation_velocity
        rotational_vector = rotational_velocity
        angular_momentum_unshape = np.matmul(self.inertia_tensor, rotational_vector)
        angular_momentum = np.reshape(angular_momentum_unshape, (3,))
        return angular_momentum

    def linear_momentum(self) -> np.array:
        """Calculates linear momentum (global system of coordinates).

        Args:
            None

        Returns:
            np.array
        """

        return self.mass * self.linear_velocity

    def kinetic_energy(self) -> float:
        """Method calculates kinetic energy

        Args:
            None

        Returns:
            float : energy
        """
        rotational_energy = self.rotational_energy()
        kin_en_lin = self.linear_energy()
        return rotational_energy + kin_en_lin

    def inertia_tensor_rotation(self):
        pass

    def update(self) -> None:
        """Updates every element inside the class."""

        self.__vertex_position()
        edge = self.edges_change()
        self.set_edges(edge)
        self.update_faces()
        self.vector_surface()

