import numpy as np
from body import Body
from particle import Particle
from scipy.integrate import odeint

class Space:
    """Space class controls and updates every object that is subscribed
    to it. Updates positions and velocity, if an object does not exist
    anymore, then it will unsuscribe it.

    Uses solve_ivp from Scipy to integrate positions of every
    object.
    """
    vel0_coll_resp_body = np.zeros((6,))
    vel0_coll_resp_particle = np.zeros((3,))
    __EPS = 1.e-15
    __minimum = 5.e-4
    __minimum_edge = 5.e-4
    def __init__(self, init_time = 0.0, gravity : float = 9.88,
                 _wp = np.float64):

        self.gravity = _wp(gravity)
        self.object_subscribed = np.array([])
        self.time = init_time
        self.size_shepre_space = 1.e1
        self._wp = _wp
        self.collision_queue = {}



    def subscribe_objects_into_space(self, object_to_subscribe) -> None:
        """Method subscribes objects into instance to iterate and update

        Args:
            object_to_subscribe : object

        Returns:
            None
        """
        self.object_subscribed = np.append(self.object_subscribed, object_to_subscribe)
        for index, objects in enumerate(self.object_subscribed):
            for index_2, objects_2 in enumerate(self.object_subscribed):
                if index != index_2:
                    key = f"{objects}{objects_2}"
                    self.collision_queue[key] = [False,0,False,0]



    def unsuscribe_objects(self, object_to_unsuscribe :\
                           list or tuple or np.ndarray or object = None) -> None:
        """Function removes elements subscribed to Space instance.
        It can recive iterable object with objects to unsuscribe

        Args:
            object_to_unsuscribe : elements to delete in space instance

        Returns:
            None
        """

        if self.object_subscribed.size>0:
            res = np.apply_along_axis(lambda x : [el.get_state() for el in x] ,0, self.object_subscribed)
            self.object_subscribed = np.delete(self.object_subscribed, res)

    def get_objects_in_space(self):
        """Getter method. Gets objects in space

        Args:
            None

        Returns:
            np.array; elements in space.
        """
        return self.object_subscribed


    def box_limit(self):
        for object_x in self.object_subscribed:
            object_x_position = object_x.get_position()
            radius = np.sqrt(np.dot(object_x_position, object_x_position))
            if radius >= self.size_shepre_space:
                vel = object_x.get_velocity()
                object_x.set_velocity(vel)
                new_pos = object_x_position / radius * self.size_shepre_space *0.999999
                object_x.set_position(new_pos)

    def __str__(self):
        """Prints every attribute of Space class"""
        string = ""
        for item, values in self.__dict__.items():
            string = f"{string}{item} {values} {type(values)}\n"

        print(self.__dict__)
        return string

    def collision_detect(self, first : Body or Particle,
                         second : Body or Particle) -> bool:
        """Method evaluates conditions for collisions and apply changes
        on velocity and angular velocity of objects suscribed.

        Args:
            first : first object to test.
            second : second object to test.

        Returns:
            bool
        """
        collision = False
        if isinstance(first, Particle) and isinstance(second, Particle):
            sphere_first = first.get_position()
            sphere_second = second.get_position()
            radius_first = first.get_size()
            radius_second = second.get_size()
            radial_sep = sphere_second - sphere_first
            radial_distance = np.sqrt(np.dot(radial_sep, radial_sep))
            radius = radius_first + radius_second
            if radial_distance <= radius:
                self.momentum_collision_partice_particle(first, second)
                collision = True


        elif ((isinstance(first, Particle) and isinstance(second, Body)) |
              (isinstance(second, Particle) and isinstance(first, Body))):
            if isinstance(first, Particle):
                particle_instance = first
                body_instance = second
            else:
                particle_instance = second
                body_instance = first
            #if self.__overlap_box(body_instance, particle_instance):
                #print("BODY - PARTICLE COLLISION")
            collision_bool, face_index, _ = self.point_to_face_collision(body_instance, particle_instance)
            if collision_bool:
                self.momentum_collision_heterogeneous(body_instance, particle_instance,
                                                       face_index )

            #collision = True
        elif isinstance(first, Body) and isinstance(second, Body):
            if self.__overlap_box(first, second):
#                print("BODY-BODY COLLISION")
                array_of_obj = np.array([[first,second],[second,first]])
                for first, second in array_of_obj:
                    collision_bool, face_index, vertex_index = self.point_to_face_collision(first, second)
                    if collision_bool:
                        print("VERTEX COLLISION")
                        self.momentum_collision_heterogeneous(first, second,
                                                           face_index, vertex_index=vertex_index)

                    collide_edge, face_index, position_of_impact = self.edge_to_edge_collision(first, second)
                    if collide_edge:
                        print("EDGE COLLISION")
                        self.momentum_collision_heterogeneous(first, second, face_index,
                                                                      edge_position=position_of_impact)


#                    if collide_edge or collision_bool: break
        return collision

    def edge_to_edge_collision(self, first : Body, second : Body) -> tuple:
        """Method evaluates collision between edges. Implemented for
        parallel edges and non perpendicular

        Args:
            first : first object to tests.
            second : second object to test.

        Returns:
            tuple : collide (bool), face_index (int), position_of_impact (np.array)

        """
        face_ok_to_collide = first.get_faces()
        edges_to_collide = second.get_edges()
        point_of_impact = None
        face_index = 0
        collide = False
        for index, face in enumerate(face_ok_to_collide):
            for edge in edges_to_collide:
                edge_of_face = face.get_edges()
                for ed in edge_of_face:
                    collide, point_of_impact = self.__linear_system_solver(ed, edge, first, second)
                    face_index = index
                    if collide: break

            if collide:
                break

        return collide, face_index, point_of_impact


    def __linear_system_solver(self, side_1 : np.array, side_2: np.array,
                               first : Body, second : Body) -> (bool, np.array):
        """Method evaluates if edges are in collision state, if so, calculates
        the point of impact.

        Args:
            side_1 : edge [base, point].
            side_2 : edge [base, point].
            first : first object to test.
            second : second object to test.

        Returns:
            tuple : (bool, point of impact)

        """


        key = f"{first}{second}"
        result = (False, np.zeros((3,)))
        side_1_vector = side_1[1] - side_1[0]
        side_1_norm = np.linalg.norm(side_1_vector)
        side_1_versor = side_1_vector / side_1_norm

        side_2_vector = side_2[1] - side_2[0]
        side_2_norm = np.linalg.norm(side_2_vector)
        side_2_versor = side_2_vector / side_2_norm
        sides_cross = np.cross(side_1_vector, side_2_vector)
        sides_cross_norm = np.linalg.norm(sides_cross)
        if sides_cross_norm <= self.__EPS and self.collision_queue[key][1] == 0:
            self.collision_queue[key][0] = True
            self.collision_queue[key][1] = sides_cross_norm
        if self.collision_queue[key][0]:
            if sides_cross_norm < self.collision_queue[key][1]:
                self.collision_queue[key][1] = 0
                self.collision_queue[key][0] = False
            #Parallel
            base_vector_1 = side_1[0]
            base_vector_2 = side_2[0]
            proyection = np.dot(base_vector_2 - base_vector_1, side_1_vector)
            Q_point = base_vector_1 + side_1_versor*proyection
            Q_point_to_base_2 = side_2[0] - Q_point
            distance = np.linalg.norm(np.cross(side_1_versor, Q_point_to_base_2))
            if distance <= self.__minimum_edge:
                inside = False
                for i in np.arange(2):
                    for j in np.arange(2):
                        condition_1 = np.linalg.norm(side_1[i] - side_2[j])
                        condition_2 = np.linalg.norm(side_1[j] - side_2[j])
                        inside_1 = condition_1<=side_1_norm and condition_1 <=side_2_norm
                        inside_2 = condition_2<=side_1_norm and condition_2 <=side_2_norm
                        if inside_1:
                            PoI = condition_1*side_1_versor + side_1[0]
                            inside = True
                            result = (inside, PoI)
                        elif inside_2:
                            PoI = condition_2*side_2_versor + side_2[0]
                            inside = True
                            result = (inside, PoI)
                        if inside: break
                    if inside: break
                pass
        else:
            P_vector = side_2[0] - side_1[0]
            numerator = np.abs(np.dot(P_vector, sides_cross))
            denominator = sides_cross_norm
            distance = numerator/denominator
            if distance <= self.__minimum_edge and self.collision_queue[key][3] == 0 :
                self.collision_queue[key][2] = True
                self.collision_queue[key][3] = distance
            if self.collision_queue[key][2]:
                if distance < self.collision_queue[key][3]:
                    self.collision_queue[key][3] = 0
                    self.collision_queue[key][2] = False
                vector_to_proyect_1 = side_2[0] - side_1[0]
                vector_to_proyect_2 = side_1[0] - side_2[0]
                proyection_1 = np.dot(vector_to_proyect_1, side_1_vector)
                proyection_2 = np.dot(vector_to_proyect_2, side_2_vector)
                proyection_condition_1 = (proyection_1>0) and (proyection_1<=side_1_norm)
                proyection_condition_2 = (proyection_2>0) and (proyection_2<=side_2_norm)
                if proyection_condition_1 and proyection_condition_2:
                    point_of_impact = proyection_1 * side_1_versor + side_1[0]
                    result = (True, point_of_impact)
        return result


    def __overlap_box(self, this : Body, other : Body or Particle) -> bool:
        """Method evaluates overlap between hit boxes. If overlaping, then
        other methods takes place.

        Args:
            this : first object to get hit box.
            other : second object to get hit box.

        Returns:
            bool

        """

        this_position = this.get_position()
        other_position = other.get_position()
        relative_direction_this_object = other_position - this_position
        #Collision box test:

        this_collision_box_global = this.collision_box()
        if isinstance(other, Body):
            other_collision_box_global = other.collision_box()
            difference_collision_box_global = np.array([this_collision_box_global[i]-other_collision_box_global[i]
                                                        for i in np.arange(len(this_collision_box_global)) ])
        else:
            difference_collision_box_global = np.array([this_collision_box_global[i]-other_position
                                                        for i in np.arange(len(this_collision_box_global)) ])

        overlap = np.apply_along_axis(this._direction, 1, difference_collision_box_global,
                                      relative_direction_this_object)
        #overlap = np.array([this._direction(diff_coll_box, relative_direction_this_object)
        #                    for diff_coll_box in difference_collision_box_global])
        overlap_result = np.sum(overlap)

        return bool(overlap_result)

    def point_to_face_collision(self, body_object : Body,
                                particle_object: Particle or Body) -> tuple:

        """Method evaluates if a point is hitting a face (Plane class).

        Args:
            body_object : Body class object to get faces.
            particle_object : Particle class to test collision

        Returns:
            tuple : bool, index of face, point of impact
        """

        #V.2 of function
        vertex_number = 0
        if isinstance(particle_object, Particle):
            array_of_positions = np.array([particle_object.get_position()])
            particle_radius = particle_object.get_size()
        elif isinstance(particle_object, Body):
            array_of_positions = particle_object.get_vertex_position()
            particle_radius = self.__minimum
        else:
            raise NotImplementedError

        body_faces = body_object.get_faces()
        for vertex_index, particle_position in enumerate(array_of_positions):
            for index,face in enumerate(body_faces):
                on_plane = False
                collision = False
                face_versor = face.get_surface_vectors()
                face_position = face.get_position()
                center_face_to_particle = particle_position - face_position
                distance_face_point = np.abs(np.dot(center_face_to_particle, face_versor))
                if distance_face_point <= particle_radius:
                    on_plane = True
                if on_plane:
                    bool_array_dot_product = np.array([], dtype=bool)
                    vertex_per_face = face.get_vertex_position()
                    for vertex in vertex_per_face:
                        vertex_to_particle = particle_position - vertex
                        if np.dot(center_face_to_particle, vertex_to_particle)>self.__EPS:
                            dot_result = np.array([True], dtype = bool)
                        else:
                            dot_result = np.array([False], dtype = bool)
                        bool_array_dot_product = np.append(bool_array_dot_product, dot_result)
                    if np.sum(np.array(bool_array_dot_product)) < len(vertex_per_face):
                        collision = True
                        face_index = index
                        vertex_number = vertex_index
                if collision:
                    return True,face_index,vertex_number
        return False, 0, 0

    def body_to_body_faces(self, first : Body, second : Body) -> tuple:
        """Method evaluates if faces may collide.

        Args:
            first : Body to get faces.
            second : body to get faces.

        Returns:
            tuple : (np.array: first faces in collision,
                    np.array: second faces in collision)

        """
        first_position = first.get_position()
        second_position = second.get_position()
        difference_positions = second_position - first_position
        first_faces = first.get_faces()
        second_faces = second.get_faces()

        first_face_mask  = [True if np.dot(face.get_surface_vectors(),
                                           difference_positions)>=0. else False
                            for face in first_faces]
        second_face_mask  = [True if np.dot(face.get_surface_vectors(),
                                           -difference_positions)>=0. else False
                            for face in second_faces]

        first_collision_faces = first_faces[first_face_mask]
        second_collision_faces = second_faces[second_face_mask]
        return first_collision_faces, second_collision_faces

    def momentum_collision_heterogeneous(self, first : Body,
                                         second : Particle or Body ,
                                         face_index_first : int,
                                         face_index_second : int = None,
                                         vertex_index = None,
                                         edge_position = None,
                                         elastic = True) -> None:

        """Function to calculates new velocity of body center and
        angular velocity after collision

        Args:
            first : Body object to collide.
            second : Body or Particle object to collide with first.
            face_index_first : int; sets index of face impact on first.
            face_index_second : int; sets index of face impact on second (NotImplemented).
            vertex_index : int; if is not None, vertex defines point of impact on first.
            edge_position : np.array; if is not None; sets point of impact.
            elastic : bool; True -> elastic collision; False -> inelastic (NotImplemented)


        """
        if elastic:
            e = 1
        else:
            e = 0

        body_position = first.get_position()
        body_velocity = first.get_velocity()
        body_angular_rotation = first.get_rotation_velocity()
        body_impact_face = first.get_faces()[face_index_first]
        body_normal_face_versor = body_impact_face.get_surface_vectors()
        body_inverse_inertia_tensor = first.get_inertia_tensor_inverse()
        body_mass = first.get_mass()


        particle_velocity = second.get_velocity()
        particle_mass = second.get_mass()
        if isinstance(second, Particle):
            particle_position = second.get_position()
            particle_size = second.get_size()
            #body_normal_face_versor+=particle_position
            PoI_2 = particle_position - body_normal_face_versor*particle_size
            PoI = PoI_2  - body_position
        elif isinstance(second, Body) and edge_position is None:
            vertex_positions_array = second.get_vertex_position()
            particle_position = vertex_positions_array[vertex_index]
            #body_normal_face_versor+=particle_position
            PoI_2 = (particle_position - second.get_position())
            PoI = particle_position - body_position

        if edge_position is not None:
            PoI_2 = edge_position - second.get_position()
            PoI = edge_position - first.get_position()

        particle_angular_rotation = second.get_rotation_velocity()
        particle_inverse_inertia_tensor = second.get_inertia_tensor_inverse()
            # r x mv_i + I_w_i = I_w_f + r x mv_f
            # mv_i + MV_i = mv_f + MV_f
            # 0.5*(mv² + MV² + wIw)

        relative_velocity = particle_velocity+ np.cross(particle_angular_rotation, PoI_2) -\
            (body_velocity + np.cross(body_angular_rotation, PoI))
        vel_r = np.dot(relative_velocity, body_normal_face_versor)
        r_x_n = np.cross(PoI, body_normal_face_versor)
        I_i_r_x_n = np.matmul(body_inverse_inertia_tensor, r_x_n)
        I_i_r_x_n = np.reshape(np.array(I_i_r_x_n),(3,))
        r_x_n_2 = np.cross(PoI_2, body_normal_face_versor)
        I_i_r_x_n_2 = np.matmul(particle_inverse_inertia_tensor, r_x_n_2)
        I_i_r_x_n_2 = np.reshape(np.array(I_i_r_x_n_2),(3,))

        denom_1 = 1./body_mass + 1./particle_mass
        denom_2_1 = np.cross(I_i_r_x_n, PoI)
        denom_2 = np.dot(denom_2_1, body_normal_face_versor)
        denom_3_1 = np.cross(I_i_r_x_n_2, PoI_2)
        denom_3 = np.dot(denom_3_1, body_normal_face_versor)

        numerator = -(1+e)*vel_r
        denominator = denom_1 + denom_2 + denom_3


        impulse = numerator/denominator
        print(impulse)
        b_new_a_rot = body_angular_rotation - I_i_r_x_n * impulse
        p_new_a_rot = particle_angular_rotation + I_i_r_x_n_2 * impulse
        p_new_velocity = particle_velocity + impulse*body_normal_face_versor/particle_mass
        b_new_velocity = body_velocity - impulse*body_normal_face_versor / body_mass

        first.set_velocity(b_new_velocity )
        first.set_rotation_velocity(b_new_a_rot)
        second.set_velocity(p_new_velocity)
        second.set_rotation_velocity(p_new_a_rot)


    def momentum_collision_partice_particle(self, first, second) -> None:
        """Method defines elastic collision between particles. Sets new
        velocity of each.

        Args:
            first : Particle to collide.
            second : Particle to collide.

        Returns:
            None
        """
        first_velocity = first.get_velocity()
        second_velocity = second.get_velocity()
        first_mass = first.get_mass()
        second_mass = second.get_mass()
        system_mass = first_mass + second_mass

        v_cm = (first_velocity*first_mass + second_velocity*second_mass)/system_mass

        first_velocity = first_velocity - v_cm
        second_velocity = second_velocity - v_cm

        first_new_velocity = (first_velocity*(first_mass-second_mass) +
                             2.*second_velocity*second_mass)/system_mass

        second_new_velocity = (second_velocity*(second_mass-first_mass) +
                              2.*first_velocity*first_mass)/system_mass

        first_new_velocity = first_new_velocity
        second_new_velocity = second_new_velocity
        first.set_velocity(first_new_velocity + v_cm)
        second.set_velocity(second_new_velocity + v_cm)

    def force(self, y, t, mass) -> np.array:
        """Defines derivative of velocity. f = m * a -> f = m * dv/dt

        Args:
            y : np.array; initial conditions.
            t : float; independent variable

        Returns:
            np.array

        """
        f = np.array([0,0,np.cos(y[2])])
        dv_dt = f/mass
        return dv_dt

    def posi(self, y, t, vel0):
        """Defines derivative of position.

        Args:
            y : np.array; initial conditions of y.
            t : float; independent variable
            vel0 : initial conditions

        Returns:
            np.array

        """
        dr_dt = vel0
        return dr_dt

    def integrate(self, step : float) -> None:
        """Method integrates velocity and position over time. Sets new
        arguments for every object subscribed in Space instance.

        Args:
            step : float; independent variable step

        Returns:
            None

        """
        for obj in self.object_subscribed:
            t = np.linspace(start=self.time, stop=self.time + step, num=501)
            if isinstance(obj, Body):
                vel0 = np.append(obj.get_velocity(), obj.get_rotation_velocity())
                pos = np.append(obj.get_position(), obj.get_angular_position())
            else:
                vel0 = obj.get_velocity()
                pos = obj.get_position()

            r = odeint(func=self.posi, y0 = pos, t=t, args=(vel0,))
            if isinstance(obj, Body):
                obj.set_position(r[-1][:3])
                obj.set_angular_position(r[-1][3:])
            else:
                obj.set_position(r[-1])

    def update(self, time : float) -> int:
        """Updates time, position and velocity for every object using self.integrate method.
        Evaluates collisions between object and execute updates method of each.

        Args:
            time : float; time step of independent variable.

        Returns:
            int : sum of collisions
        """

        collision = np.array([self.collision_detect(obj, sec)
                      for index,obj in enumerate(self.object_subscribed)
                      for sec in self.object_subscribed[index:]
                      if obj!= sec])

        #np.array([obj.update() for obj in self.object_subscribed])
        total_collision = np.sum(collision)
        self.integrate(time)
        self.time += time
        np.array([obj.update() for obj in self.object_subscribed])

        return total_collision


