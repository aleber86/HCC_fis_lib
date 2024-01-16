import numpy as np
from object_class_module import render
from body import Body
from particle import Particle
from cube import Cube
from cilinder import Cilinder
from scipy.integrate import odeint

class Space:
    """Space class controls and updates every object that is subscribed
    to it. Updates positions and velocity, if an object does not exist
    anymore, then it will unsuscribe it.

    Uses solve_ivp from Scipy to integrate positions of every
    object.
    """
    __EPS = 1.e-15
    def __init__(self, init_time = 0.0, gravity : float = 9.88,
                 air_res : float = 0.5, _wp = np.float64):

        self.gravity = _wp(gravity)
        self.object_subscribed = np.array([])
        self.time = init_time
        self.air = _wp(air_res)
        self.size_shepre_space = 1.e26
        self.collision_max_distance = 1.e-5
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
                object_x.set_velocity(vel*0)
                if radius>0:
                    new_pos = object_x_position / radius * self.size_shepre_space *0.99999
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
            self.point_to_face_collision(body_instance, particle_instance)

            #collision = True
        elif isinstance(first, Body) and isinstance(second, Body):
            if self.__overlap_box(first, second):
                self.face_to_face_collision(first, second)
                print("BODY-BODY COLLISION")
                collision = True
        return collision


    def __overlap_box(self, this : Body, other : Body or Particle):

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
        overlap_result = np.sum(overlap)

        return bool(overlap_result)

    def point_to_face_collision(self, body_object, particle_object):
        #V.2 of function
        particle_position = particle_object.get_position()
        particle_radius = particle_object.get_size()
        body_faces = body_object.get_faces()
        for index,face in enumerate(body_faces):
            on_plane = False
            collision = False
            face_versor = face.get_surface_vectors()
            face_position = face.get_position()
            center_face_to_particle = particle_position - face_position
            distance_face_point = np.abs(np.dot(center_face_to_particle, face_versor))
            if distance_face_point <= particle_radius + self.__EPS:
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
            if collision:
                self.momentum_collision_heterogeneous(body_object, particle_object,
                                                      particle_position, face_index )

    def face_to_face_collision(self, first : Body, second : Body):
        first_versors = np.array([face.get_surface_vectors() for face in first.get_faces()])
        second_versors = np.array([second.get_surface_vectors() for face in second.get_faces()])
        print(first_versors)
        print(second_versors)

    def momentum_collision_heterogeneous(self, first : Body,
                                         second : Particle or Body ,
                                         position : np.array,
                                         face_index_first : int, face_index_second : int = None,
                                         elastic = True):

        """Function to Calculate momentum after collision"""
        if elastic:
            e = 1
        else:
            e = 0

        if face_index_second is None:
            """Body - Particle momentum transfer"""
            body_position = first.get_position()
            body_velocity = first.get_velocity()
            body_angular_rotation = first.get_rotation_velocity()
            body_impact_face = first.get_faces()[face_index_first]
            body_normal_face_versor = body_impact_face.get_surface_vectors()
            body_inverse_inertia_tensor = first.get_inertia_tensor_inverse()
            body_mass = first.get_mass()


            particle_position = position
            particle_velocity = second.get_velocity()
            particle_mass = second.get_mass()

            #body_new_velocity = body_velocity  -  j_r * body_normal_face_versor / body_mass
            #particle_new_velocitty = particle_velocity + j_r * body_normal_face_versor / particle_mass

            PoI = particle_position - body_position
            #Position of impact in local body reference system: PoI = particle_position - body_position
            #body_new_angular_rotation = body_angular_rotation - j_r * body_inverse_inertia_tensor * (PoI)

            body_v_rot_lin = body_velocity + np.cross(body_angular_rotation, PoI)

            relative_velocity = particle_velocity - body_v_rot_lin

            #relative_new_velocity · versor  = - e * (relative_velocity · versor)
            denom = np.matmul(body_inverse_inertia_tensor,np.cross(PoI, body_normal_face_versor))
            last_term = np.cross(denom, PoI)
            last_term_dot = np.dot(last_term, body_normal_face_versor)
            impulse_relative = -(1+e)* np.dot(relative_velocity,body_normal_face_versor)/ \
                                (1./body_mass + 1./particle_mass + last_term_dot)

            body_new_velocity = body_velocity  -  impulse_relative * body_normal_face_versor / body_mass
            particle_new_velocitty = particle_velocity + impulse_relative * body_normal_face_versor / particle_mass

            PoI_x_versor = np.cross(PoI, body_normal_face_versor)
            body_new_angular_rotation = body_angular_rotation - impulse_relative \
                                        * np.matmul(body_inverse_inertia_tensor,PoI_x_versor)

            first.set_velocity(body_new_velocity)
            first.set_rotation_velocity(body_new_angular_rotation)
            second.set_velocity(particle_new_velocitty)


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

        first_new_velocity = first_new_velocity + v_cm
        second_new_velocity = second_new_velocity + v_cm
        first.set_velocity(first_new_velocity)
        second.set_velocity(second_new_velocity)

        print(first_new_velocity, second_new_velocity)

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
            t = np.linspace(start=self.time, stop=self.time + step, num=101)
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

        collision = np.array([self.collision_detect(obj, sec)
                      for index,obj in enumerate(self.object_subscribed)
                      for sec in self.object_subscribed[index:]
                      if obj!= sec])

        np.array([obj.update(time) for obj in self.object_subscribed])
        self.box_limit()
        self.integrate(time)
        self.time += time
        total_collision = np.sum(collision)
        return total_collision

if __name__ == '__main__':
    np.random.seed(456791)
    dim = 1
    space_instance = Space()
    particles = [Particle(1.e-3,1,[.0,0.5,-1],[0,0,10]),
                 #Particle(1.e-3,1,[.1,0.1,-1],[0,0,2]),
                 #Particle(1.e-2, 2, [1,0,0], [-1,0,0]),
                 Particle(1.e-3, 1, [0,-0.5,1], [0,0,-10]),
                 Cube(1,20,0,[0.,0.,0.],[0.,0.,0.],[0.,0.,0],[0., 0.,0.], False)]
    space_instance.subscribe_objects_into_space(particles)
    step = 0.01
    condition = True
    time_start = 0.0
    counter = 0
    collision = 0
    to_text = ""
    total_energy = np.sum(np.array([obj.kinetic_energy() for obj in space_instance.get_objects_in_space()]))
    print(np.array([obj.kinetic_energy() for obj in space_instance.get_objects_in_space()]))
    while condition:
        time_start += step
        if counter % 4 == 0:
            render(space_instance.get_objects_in_space(), 1)
            total_energy = np.sum(np.array([obj.kinetic_energy() for obj in space_instance.get_objects_in_space()]))
            print(np.array([obj.kinetic_energy() for obj in space_instance.get_objects_in_space()]))
            print(f"Total energy: {total_energy}")
        collision = space_instance.update(step)
        counter+=1
