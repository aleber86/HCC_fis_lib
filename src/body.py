import numpy as np
from particle import Particle

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
        self._wp = _wp
        self.mass = _wp(mass)
        self.friction = _wp(friction)
        self.position = np.array(init_pos, dtype = _wp)
        self.angular_position = np.array(init_angular, dtype=_wp)
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
        self.__vertex_position(0.0)
        self.edges = self.edges_change()
        self.inertia_tensor = np.zeros((3,3), dtype = _wp)
        self.axial_vectors_to_faces = None
        self.state = True
        self.surface_vector = None
        self.change_variable_state = True
        self._tolerance = _wp(self._tolerance)

    def update_faces(self):
        if self.faces is not None:
            offset = 0
            for index,face in enumerate(self.faces):
                vertex_len = len(face.get_vertex_position())
                self.faces[index].set_vertex_position(self.global_vertex[offset:offset+vertex_len])
                self.faces[index].vector_surface()
                rot = self.faces[index].get_angular_position()
                self.faces[index].set_position(np.array(self.global_vertex[offset]
                                                - self.faces[index].get_vertex_position_local()[0], dtype = self._wp))
                self.faces[index].set_angular_position(rot)
                offset += vertex_len


    def change_status(self, variable_change, variable_to_change):
        """It determines if phase state variables change. If not
        there will be no rotations needed. Reduce computational
        usage"""

        state = np.sum(np.abs(variable_change - variable_to_change))
        if state > self._tolerance:
            self.change_variable_state = True
        else:
            self.change_variable_state = False

    def volume_calc(self, samples_quant = 5000):
        """Function implemented by Monte Carlo simulation as in CAD systems
        to calculate volume of 3D bodys"""
        counter = np.array([0])
        samples = samples_quant
        position_of_faces = np.array([faces.get_position() for faces in self.faces])
        local_surface_vector = np.array([faces.get_surface_vectors() for faces in self.faces])
        max_x , max_y, max_z = self.hit_box_local[-3]
        min_x , min_y, min_z = self.hit_box_local[0]
        volume_total_box = np.prod(np.array([np.abs(sides[1]-sides[0])
                                for sides in zip(self.hit_box_local[-3], self.hit_box_local[0])]
                                            , dtype = self._wp))
        array_random_points = np.array([np.array([np.random.uniform(min_x, max_x),
                                np.random.uniform(min_y, max_y),
                                np.random.uniform(min_z, max_z)])
                                for _ in np.arange(samples)] , dtype = self._wp)
        self.array_point = array_random_points
        np.apply_along_axis(self.__point_calculation, 1, array_random_points,
                                       local_surface_vector, position_of_faces, counter)
        print(counter)
        total_inside = counter[0]
        print(volume_total_box, total_inside)
        return total_inside/samples*volume_total_box

    def __point_calculation(self, point, local_surface_vector, position_of_faces, counter):
        bool_inside = np.array([np.dot(-point + pos, face_vector_surface)>= 0.
                                for pos, face_vector_surface in zip(position_of_faces,local_surface_vector)], dtype = bool)

        if False in bool_inside:
            pass
        else:
            counter[0] = counter[0] + 1

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

    def get_friction(self):
        return self.friction

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def set_axial_vector_to_faces(self, vectors_to_center : np.array):
        self.axial_vectors_to_faces = vectors_to_center

    def set_angular_position(self, angular_position : np.array):
        self.angular_position = angular_position

    def set_edges(self, edg):
        self.edges = edg

    def set_inertia_tensor(self, inertia_tensor):
        self.inertia_tensor = inertia_tensor

    def set_linear_velocity(self, velocity):
        self.change_status(velocity, self.linear_velocity)
        self.linear_velocity = velocity

    def set_position(self, position):
        self.change_status(position, self.position)
        self.position = position

    def set_rotation_velocity(self, rotation):
        self.change_status(rotation, self.rotation_velocity)
        self.rotation_velocity = rotation

    def set_vertex_position(self, position_of_vertex : np.array):
        self.global_vertex = position_of_vertex

    def _vector_rotation(self, vector_to_rotate, angular_deviation, origin):
        """Rotation matrix = R_z(alpha)*R_y(beta)*R_x(gamma) | RotMat * VECTOR"""

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


    def __vertex_position(self, delta_time : float = 0.0, axis : np.array or None = None):
        """Updates center position and vertex position on global system
            If axis = None, the body rotates over an axis thus it center (not the mass center,
            body defined center by instant position)"""

        delta_time = self._wp(delta_time)
        linear_velocity = self.get_velocity()
        rotational_velocity = self.get_rotation_velocity()
        center_position = self.get_position()
        angular_deviation = self.get_angular_position()
        center_update_position = self.__change_by_time(center_position , linear_velocity, delta_time)
        angular_update_direction = self.__change_by_time(angular_deviation , rotational_velocity, delta_time)
        if axis is None:
            origin = np.zeros(center_position.shape)
        else:
            origin = axis

        vertex_update_position = np.apply_along_axis(self._vector_rotation, 1, self.local_vertex,
                                                     angular_update_direction, origin )
        vertex_update_position = np.array(vertex_update_position+center_update_position)
        self.set_position(np.array(center_update_position, dtype = self._wp))
        self.set_angular_position(np.array(angular_update_direction, dtype=self._wp))
        self.set_vertex_position(vertex_update_position)

        hit_box_update_position = np.apply_along_axis(self._vector_rotation, 1, self.hit_box_local,
                                                     angular_update_direction, origin )

        hit_box_update_position = np.array(hit_box_update_position+center_update_position)
        self.hit_box_global = hit_box_update_position


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

    def collision_detect(self, other):
        """Two step detection function.
        First detection: function test collision_box overlap. If true,
        second detection: collision test using faces.
        """
        if isinstance(other, Body):
            this_position, other_position =  self.get_position(), other.get_position()
        else:
            this_position, other_position = self.get_position(), other.get_position()
        relative_direction_this_object = other_position - this_position
        #Collision box test:

        this_collision_box_global = self.collision_box()
        other_collision_box_global = other.collision_box()
        difference_collision_box_global = np.array([this_collision_box_global[i]-other_collision_box_global[i]
                                                    for i in np.arange(len(other_collision_box_global)) ])
        overlap = np.apply_along_axis(self.__direction, 1, difference_collision_box_global,
                                      relative_direction_this_object)
        overlap_result = np.sum(overlap)

        #If collision box overlaps
        #and if `other` is is instance of `Body`:
        if overlap_result!= 0 and isinstance(other, Body):
            print("BODY-BODY COLLISION")
            face_vector_surface_this_object = self.get_surface_vectors()
            face_vector_surface_other_object = other.get_surface_vectors()
            relative_direction_other_object = - relative_direction_this_object
            #Returns mask for the face_vector_surface:
            mask_of_this = np.apply_along_axis(self.__direction,1, face_vector_surface_this_object,
                                relative_direction_this_object)

            mask_of_other = np.apply_along_axis(self.__direction,1, face_vector_surface_other_object,
                                relative_direction_other_object)

            face_to_collide_this = self.faces[mask_of_this]
            face_to_collide_other = other.faces[mask_of_other]
            #np.array([print(faces.get_vertex_position()) for faces in face_to_collide_this])
            #np.array([print(faces.get_vertex_position()) for faces in face_to_collide_other])
            #print(face_to_collide_other)
            #print(mask_of_this, mask_of_other)
            #print(face_vector_surface_other_object)
            #print(face_vector_surface_this_object)
        elif overlap_result!= 0 and isinstance(other, Particle):
            print("PARTICLE - BODY COLLISION")

    def __direction(self, face_vector, vector_to_proyect):
        """A · B = |A||B| Cos(b). If A · B >= 0, then collision could occurre"""
        condition = False
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

    def update(self, delta_time : float):
        delta_time = self._wp(delta_time)
        if self.change_variable_state:
            self.__vertex_position(delta_time)
            edge = self.edges_change()
            self.set_edges(edge)
            self.vector_surface()
            self.update_faces()

