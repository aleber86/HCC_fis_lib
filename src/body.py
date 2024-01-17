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
        self.angular_position = np.fmod(np.array(init_angular, dtype=_wp), 2.*np.pi)
        self.linear_velocity = np.array(init_vel, dtype = _wp)
        self.rotation_velocity = np.array(init_rot, dtype = _wp)
        self.destructive = destructive
        self.edges_indexes = np.array(edges_indexes, dtype = np.int32)
        self.volume = 1.0 #Implemented by self.volume_calc funcion
        self.density = 1.0 #Implemented by self.density_calc function
        self.hit_box_local = np.array([0,0,0])
        self.hit_box_global = np.array([0,0,0])
        self.local_vertex = _wp(vertex)
        self.global_vertex = 0.0
        self.array_point = np.array([[0,0,0]])
        self.__collision_box()
        self.__vertex_position()
        self.edges = self.edges_change()
        self.inertia_tensor = np.zeros((3,3), dtype = _wp)
        self.inertia_tensor_inverse = np.zeros((3,3), dtype = _wp)
        self.axial_vectors_to_faces = None
        self.state = True
        self.surface_vector = None
        self.change_variable_state_position = True
        self.change_variable_state_angular = True
        self.change_variable_state_rotation_v = True
        self.change_variable_state_vel = True
        self._tolerance = _wp(self._tolerance)
        self.update_faces()

    def update_faces(self):
        if self.faces is not None:
            offset = 0
            for index,face in enumerate(self.faces):
                vertex_len = len(face.get_vertex_position())
                self.faces[index].set_vertex_position(self.global_vertex[offset:offset+vertex_len])
                offset += vertex_len
                self.faces[index].vector_surface()


    def change_status(self, variable_change, variable_to_change):
        """It determines if phase state variables change. If not
        there will be no rotations needed. Reduce computational
        usage"""

        state = np.sum(np.abs(variable_change - variable_to_change))
        if state > self._tolerance:
            flag = True
        else:
            flag = False
        return flag

    def volume_calc(self, samples_quant = 2000, n = 10):
        """Function implemented by Monte Carlo simulation as in CAD systems
        to calculate volume of 3D bodys"""
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

        return np.mean(volume_array)

    def _point_calculation(self, point, counter):
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

    def total_surface_calc(self, quant = 1000, n=10):
        if self.faces is not None:
            total_surface_per_face = np.array(
                [face.surface_calc(quant, n) for face in self.faces], dtype=self._wp)
            total_surface = np.sum(total_surface_per_face)
            return total_surface
        else:
            pass

    def density_calc(self):
        """Function sets the density of bodys. All bodys are treated as homogenous.
        rho = mass / volume
        """
        dens = self.mass/self.volume
        return dens
    def get_mass(self):
        return self.mass

    def get_edges(self):
        return self.edges

    def get_inertia_tensor(self):
        return self.inertia_tensor

    def get_inertia_tensor_inverse(self):
        return self.inertia_tensor_inverse

    def get_velocity(self):
        return self.linear_velocity

    def get_position(self):
        return self.position

    def get_angular_position(self):
        return self.angular_position

    def get_rotation_velocity(self):
        return self.rotation_velocity

    def get_vertex_position(self):
        return self.global_vertex

    def get_vertex_position_local(self):
        return self.local_vertex

    def get_edges_index(self):
        return self.edges_indexes

    def get_faces(self):
        return self.faces

    def get_friction(self):
        return self.friction

    def get_state(self):
        return self.state

    def get_offset_rotation_axis(self):
        return self.offset_rotation_axis

    def set_state(self, state):
        self.state = state

    def set_offset_rotation_axis(self, axis):
        self.offset_rotation_axis = axis

    def set_axial_vector_to_faces(self, vectors_to_center : np.array):
        self.axial_vectors_to_faces = vectors_to_center

    def set_angular_position(self, angular_position : np.array):
        self.change_variable_state_angular = self.change_status(angular_position,
                                                                self.angular_position)
        if self.change_variable_state_angular:
            self.angular_position = angular_position

    def set_edges(self, edg):
        self.edges = edg

    def set_inertia_tensor(self, inertia_tensor):
        self.inertia_tensor = inertia_tensor

    def set_inertia_tensor_inverse(self):
        self.inertia_tensor_inverse = np.linalg.inv(self.inertia_tensor)

    def set_velocity(self, velocity):
        self.change_variable_state_vel = self.change_status(velocity,
                                                            self.linear_velocity)
        if self.change_variable_state_vel:
            self.linear_velocity = velocity

    def set_position(self, position):
        self.change_variable_state_position = self.change_status(position,
                                                                 self.position)
        if self.change_variable_state_position:
            self.position = position

    def set_rotation_velocity(self, rotation):
        self.change_variable_state_rotation_v = self.change_status(rotation,
                                                                   self.rotation_velocity)
        if self.change_variable_state_rotation_v:
            self.rotation_velocity = rotation

    def set_vertex_position(self, position_of_vertex : np.array):
        self.global_vertex = position_of_vertex

    def _vector_rotation(self, vector_to_rotate, angular_deviation, origin):
        """Rotation matrix = R_z(alpha)*R_y(beta)*R_x(gamma) | RotMat * VECTOR"""

        """
        theta, psi, phi = angular_deviation
        row_1 = [np.cos(psi)*np.cos(phi)-np.cos(theta)*np.sin(phi)*np.sin(psi),
                 np.cos(psi)*np.sin(phi)+np.cos(theta)*np.cos(phi)*np.sin(psi),
                 np.sin(psi)*np.sin(theta)]
        row_2 = [-np.sin(psi)*np.cos(phi)-np.cos(theta)*np.sin(phi)*np.cos(psi),
                 -np.sin(psi)*np.sin(phi)+np.cos(theta)*np.cos(phi)*np.cos(psi),
                 np.cos(psi)*np.sin(theta)]
        row_3 = [np.sin(theta)*np.sin(phi), -np.sin(theta)*np.cos(phi), np.cos(theta)]

        rotational_matrix = np.matrix([row_1, row_2, row_3], dtype=self._wp)
        """
        gamma, beta, alpha = angular_deviation
        rotational_matrix = np.matrix([[np.cos(alpha)*np.cos(beta),
                                       np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma),
                                       np.cos(alpha)*np.sin(beta)*np.cos(gamma)+ np.sin(alpha)*np.sin(gamma)],
                                      [np.sin(alpha)*np.cos(beta),
                                       np.sin(alpha)*np.sin(gamma)+np.cos(alpha)*np.cos(gamma),
                                       np.sin(alpha)*np.sin(beta)*np.cos(gamma)-np.cos(alpha)*np.sin(gamma)],
                                      [-np.sin(beta), np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma)]],
        dtype = self._wp)
        diff = vector_to_rotate - origin

        row = diff.shape[0]
        diff_column = np.reshape(diff, (row, 1))
        vector_rotated = np.matmul(rotational_matrix, diff_column)
        return vector_rotated


    def __vertex_position(self):
        """Updates center position and vertex position on global system
            If axis = None, the body rotates over an axis thus it center (not the mass center,
            body defined center by instant position)"""
        zero_vector = np.zeros((3,), dtype = self._wp)
#        if self.change_variable_state_position:
        center_update_position = self.position
#        else:
#            center_update_position = zero_vector
        if self.offset_rotation_axis is None:
            origin = zero_vector
        else:
            origin = self.offset_rotation_axis
        #if self.faces is not None:
        #    np.array([face.set_position(face.get_position()+center_update_position)
        #                                for face in self.faces])

        if self.change_variable_state_angular:

            angular_update_direction = self.get_angular_position()
            vertex_update_position = np.apply_along_axis(self._vector_rotation, 1, self.local_vertex,
                                                         angular_update_direction, origin )
            vertex_update_position = np.array(vertex_update_position+center_update_position)
            self.set_vertex_position(vertex_update_position)
            hit_box_update_position = np.apply_along_axis(self._vector_rotation, 1, self.hit_box_local,
                                                         angular_update_direction, origin )

            hit_box_update_position = np.array(hit_box_update_position+center_update_position)
            self.hit_box_global = hit_box_update_position
        if self.faces is not None:
            positions_local= self.faces_local_position
            if self.change_variable_state_angular:
                position_new_local = np.apply_along_axis(self._vector_rotation, 1, positions_local,
                                                         angular_update_direction, origin)
            else:
                position_new_local = positions_local
            position_new_global = position_new_local
            for vec,face in zip(position_new_global,self.faces):
                vec1 = np.reshape(np.array(vec), (3,))
                face.set_position(vec1+center_update_position)
    def __change_by_time(self, change_argument, change_rate, delta_time):
        """Change on phase space"""
        changed_argument = change_argument + change_rate * delta_time
        return changed_argument


    def __define_edges(self, indexes):
        edge = np.array([self.global_vertex[indexes[0]], self.global_vertex[indexes[1]]], dtype = self._wp)
        return edge

    def edges_change(self):
        edges = np.apply_along_axis(self.__define_edges, 1, self.edges_indexes)
        return  edges

    def get_surface_vectors(self):
        return self.surface_vector

    def vector_surface(self):
        """Method of Body not defined. Every child class must overload
        the operator"""
        pass
    def _direction(self, face_vector, vector_to_proyect):
        """A · B = |A||B| Cos(b). If A · B >= 0, then collision could occurre"""
        condition = False
        if np.dot(face_vector, vector_to_proyect)>self._tolerance:
            condition = True
        return condition

    def collision_box(self):
        return self.hit_box_global

    def __collision_box(self):
        max_x = np.max(self.local_vertex[:,0])
        max_y = np.max(self.local_vertex[:,1])
        max_z = np.max(self.local_vertex[:,2])
        offset = (np.abs(max_x) + np.abs(max_y) + np.abs(max_z))/3 * 0.5
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

    def rotational_energy(self):
        rotational_velocity = self.rotation_velocity
        rotational_vector = np.reshape(rotational_velocity, (1,3))
        flat_product = np.reshape(np.matmul(rotational_vector, self.inertia_tensor), (3,))
        rot_energy = 0.5 * np.dot(flat_product,rotational_velocity)
        return rot_energy

    def linear_energy(self):
        linear_velocity = self.linear_velocity
        mass = self.mass
        lin_energy = 0.5 *mass* np.dot(linear_velocity,linear_velocity)
        return lin_energy

    def angular_momentum(self):
        rotational_velocity = self.rotation_velocity
        rotational_vector = np.reshape(rotational_velocity, (3,1))
        angular_momentum_unshape = np.matmul(self.inertia_tensor, rotational_vector)
        angular_momentum = np.reshape(angular_momentum_unshape, (3,))
        return angular_momentum

    def linear_momentum(self):
        return self.mass * self.linear_velocity

    def kinetic_energy(self):
        rotational_energy = self.rotational_energy()
        kin_en_lin = self.linear_energy()
        return kin_en_lin + rotational_energy

    def update(self, delta_time : float):
        self.__vertex_position()
        edge = self.edges_change()
        self.set_edges(edge)
        self.vector_surface()
        self.update_faces()

