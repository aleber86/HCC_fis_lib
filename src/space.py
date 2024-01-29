import numpy as np
from object_class_module import render
from body import Body
from particle import Particle
from cube import Cube
from scipy.integrate import odeint
from scipy.optimize import bisect

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
    __minimum = 2.e-2
    def __init__(self, init_time = 0.0, gravity : float = 9.88,
                 _wp = np.float64):

        self.gravity = _wp(gravity)
        self.object_subscribed = np.array([])
        self.time = init_time
        self.size_shepre_space = 1.e1
        self._wp = _wp


    def subscribe_objects_into_space(self, object_to_subscribe):
        self.object_subscribed = np.append(self.object_subscribed, object_to_subscribe)

    def unsuscribe_objects(self, object_to_unsuscribe : list or tuple or np.ndarray or object = None):
        """Function removes elements subscribed to Space instance.
        It can recive iterable object with objects to unsuscribe"""
        if self.object_subscribed.size>0:
            res = np.apply_along_axis(lambda x : [el.get_state() for el in x] ,0, self.object_subscribed)
            self.object_subscribed = np.delete(self.object_subscribed, res)

    def get_objects_in_space(self):
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

    def collision_detect(self, first, second):
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
                        self.momentum_collision_heterogeneous(first, second,
                                                           face_index, vertex_index=vertex_index)

                    collide_edge, face_index, position_of_impact = self.edge_to_edge_collision(first, second)
                    if collide_edge:
                        self.momentum_collision_heterogeneous(first, second, face_index,
                                                                      edge_position=position_of_impact)


                    if collide_edge or collision_bool: break
        return collision

    def edge_to_edge_collision(self, first : Body, second : Body, quant = 20):
        #face_ok_to_collide, _ = self.body_to_body_faces(first, second)
        face_ok_to_collide = first.get_faces()
        edges_to_collide = second.get_edges()
        point_of_impact = None
        face_index = 0
        collide = False
        for index, face in enumerate(face_ok_to_collide):
            #vertex_face = face.get_vertex_position()
            for edge in edges_to_collide:
                edge_of_face = face.get_edges()
                for ed in edge_of_face:
                    collide, point_of_impact = self.__linear_system_solver(ed, edge)
                    face_index = index
                    if collide: break

            if collide:
                break

        return collide, face_index, point_of_impact


    def __linear_system_solver(self, side_1 : np.array, side_2: np.array) -> (bool, np.array):
        result = (False, np.zeros((3,)))
        side_1_vector = side_1[1] - side_1[0]
        side_1_norm = np.linalg.norm(side_1_vector)
        side_1_versor = side_1_vector / side_1_norm

        side_2_vector = side_2[1] - side_2[0]
        side_2_norm = np.linalg.norm(side_2_vector)
        sides_cross = np.cross(side_1_vector, side_2_vector)
        sides_cross_norm = np.linalg.norm(sides_cross)

        if sides_cross_norm <= self.__EPS:
            #Parallel
            base_vector_1 = side_1[0]
            base_vector_2 = side_2[0]
            proyection = np.dot(base_vector_2 - base_vector_1, side_1_vector)
            Q_point = base_vector_1 + side_1_versor*proyection
            Q_point_to_base_2 = side_2[0] - Q_point
            distance = np.linalg.norm(np.cross(side_1_versor, Q_point_to_base_2))
            if distance <= self.__minimum:
                inside = False
                for i in np.arange(2):
                    for j in np.arange(2):
                        condition_1 = np.linalg.norm(side_1[i] - side_2[j])
                        condition_2 = np.linalg.norm(side_1[j] - side_2[j])
                        inside_1 = condition_1<=side_1_norm and condition_1 <=side_2_norm
                        inside_2 = condition_2<=side_1_norm and condition_2 <=side_2_norm
                        if inside_1:
                            PoI = condition_1*side_1_norm + side_1[0]
                            inside = True
                            result = (inside, PoI)
                        elif inside_2:
                            PoI = condition_2*side_1_norm + side_1[0]
                            inside = True
                            result = (inside, PoI)
                        if inside: break
                    if inside: break
        else:
            P_vector = side_2[0] - side_1[0]
            numerator = np.abs(np.dot(P_vector, sides_cross))
            denominator = sides_cross_norm
            distance = numerator/denominator
            if distance <= self.__minimum:
                vector_to_proyect_1 = side_2[0] - side_1[0]
                vector_to_proyect_2 = side_1[0] - side_2[0]
                proyection_1 = np.dot(vector_to_proyect_1, side_1_vector)
                proyection_2 = np.dot(vector_to_proyect_2, side_2_vector)
                proyection_condition_1 = (proyection_1>0) and (proyection_1<=side_1_norm)
                proyection_condition_2 = (proyection_2>0) and (proyection_2<=side_2_norm)
                if proyection_condition_1 and proyection_condition_2:
                    point_of_impact = proyection_1 * side_1_versor + 2*side_1[0]
                    result = (True, point_of_impact)
        return result


    def __overlap_box(self, this : Body, other : Body or Particle) -> bool:

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
                                particle_object: Particle or Body
                                ):
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

    def body_to_body_faces(self, first : Body, second : Body):
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
                                         face_index_first : int, face_index_second : int = None,
                                         vertex_index = None, edge_position = None,
                                         elastic = True):

        """Function to Calculate momentum after collision"""
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
            PoI = PoI_2  + particle_position - body_position
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
        b_new_a_rot = body_angular_rotation - I_i_r_x_n * impulse
        p_new_a_rot = particle_angular_rotation + I_i_r_x_n_2 * impulse
        p_new_velocity = particle_velocity + impulse*body_normal_face_versor/particle_mass
        b_new_velocity = body_velocity - impulse*body_normal_face_versor / body_mass


        first.set_velocity(b_new_velocity )
        first.set_rotation_velocity(b_new_a_rot)
        second.set_velocity(p_new_velocity)
        second.set_rotation_velocity(p_new_a_rot)


    def momentum_collision_partice_particle(self, first, second):
        """ELASTIC COLLISION"""
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

    def force(self, y, t, mass):
        """ f = m * a -> f = m * dv/dt"""
        f = np.array([0,0,np.cos(y[2])])
        dv_dt = f/mass
        return dv_dt

    def posi(self, y, t, vel0):
        dr_dt = vel0
        return dr_dt

    def integrate(self, step):
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

    def update(self, time):
        """Updates time, position and velocity for every object using odeint from Scipy. Time span is
        calculated with numpy linspace 10 elements, odeint calculates the interval and takes the last element
        to set new state for every object subscribed. If an element state is False then it will be unsuscribed."""

        np.array([obj.update() for obj in self.object_subscribed])
        collision = np.array([self.collision_detect(obj, sec)
                      for index,obj in enumerate(self.object_subscribed)
                      for sec in self.object_subscribed[index:]
                      if obj!= sec])

        total_collision = np.sum(collision)
        np.array([obj.update() for obj in self.object_subscribed])
        self.integrate(time)
        self.time += time
        return total_collision

if __name__ == '__main__':
    from cilinder import Cilinder
    def ang_momentum(element):

        mass = element.get_mass()
        position = element.get_position()
        velocity = element.get_velocity()
        inertia_t = element.get_inertia_tensor()
        rotation = element.get_rotation_velocity()
        angular_momentum = np.matmul(inertia_t, rotation)
        angular_momentum = np.reshape(np.array(angular_momentum), (3,))
        momentum = np.cross(position, mass*velocity) + angular_momentum
        return momentum

    def energy(element):
        mass = element.get_mass()
        velocity = element.get_velocity()
        inertia_t = element.get_inertia_tensor()
        rotation = element.get_rotation_velocity()
        angular_momentum = np.matmul(inertia_t, rotation)
        angular_momentum = np.reshape(np.array(angular_momentum), (3,))
        en = 0.5*(np.dot(angular_momentum, rotation) + mass*np.dot(velocity,velocity) )
        return en


    #file = open("data.dat", "w")
    np.random.seed(456791)
    dim = 1
    size = 0.5e0
    vel = np.array([.0,20,.0])
    #mass_1 = 1.e-25
    #mass_2 = 6
    mass_1 = 5
    mass_2 = 5
    space_instance = Space()
#    particles = [Particle(size,mass_1,[-.1+0,-4.1+0,0+.2],vel),
#                 Cube(4.,mass_2,0,[0.,0.,0.],[0.,0.,0],[0.,0,0],[0, 0.,0.], False)]
    particles = [Cube(2, mass_1, 0, [1,1,1], [0,0,0], [0.0,0.0,0], [0,0,0], False),
                 Cilinder(size,size*8,16, mass_2, 0, [1+0.3,1-2,1+0.3], [0,0,0], vel, [0,0,0], False)]
    """
    particles = np.array([Particle(size,np.random.uniform(1,20),
                                   [np.random.uniform(-0.5,0.5),
                                    np.random.uniform(-0.5,0.5),
                                    np.random.uniform(-0.5,0.5)],
                                   [np.random.uniform(-0.5,0.5),
                                    np.random.uniform(-0.5,0.5),
                                    np.random.uniform(-0.5,0.5)])
                          for _ in np.arange(100)])
    """
#    particles  = np.append(particles, [Cube(2,mass_2,0,[2,0,0],[0,0,0],[0,0,0],[0,0,0], False)])
    space_instance.subscribe_objects_into_space(particles)
    norm = np.linalg.norm(vel)
    diff = size/(norm*500)
    #diff = 0.01
    step = diff
    cond = True
    time_start = 0.0
    counter = 0
    collision = 0
    to_text = ""
    objects = space_instance.get_objects_in_space()
    np.array([obj.update() for obj in objects])
    total_energy_0_a = np.array([energy(obj) for obj in objects])
    total_energy_0 = np.sum(total_energy_0_a)
    linear_momentum = np.array([obj.linear_momentum()
                                       for obj in objects])

    total_p = np.linalg.norm(np.sum(linear_momentum,0))
    angular_momentum = np.array([ang_momentum(obj) for obj in objects])
    total_L = np.linalg.norm(np.sum(angular_momentum,0))
    while cond:
        time_start += step
        if counter%50 == 0:
            linear_momentum = np.array([obj.linear_momentum()
                                       for obj in objects])
            e_p = np.abs(np.linalg.norm(np.sum(linear_momentum,0))-total_p)/np.abs(total_p)*100
            angular_momentum = np.array([ang_momentum(obj) for obj in objects])
            e_L = np.abs(np.linalg.norm(np.sum(angular_momentum,0))-total_L)/np.abs(total_L)*100
            total_energy = np.array([energy(obj) for obj in objects])
            e_E = 100*np.abs(total_energy_0-np.sum(total_energy))/np.abs(total_energy_0)
            render(space_instance.get_objects_in_space(), counter, True)
            #print([obj.get_position() for obj in objects])
            """
            if e_p >0. or e_L>0. or e_E >0:
                print(r"Relative error % linear momentum:", \
                      f"{e_p}   in: {total_p}  fin: {np.linalg.norm(np.sum(linear_momentum,0))}")
                print(r"Relative error % angular momentum :" \
                      f"{e_L}  in: {total_L}  fin: {np.linalg.norm(np.sum(angular_momentum,0))}")
                print(r"Relative error % energy :" \
                      f"{e_E}  in :{total_energy_0}  fin: {np.sum(total_energy)}")
            """
        collision = space_instance.update(step)
        counter+=1


