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

    def collision_detect(self, first, second, radius = 1.e-7):
        collision = False
        if isinstance(first, Particle) and isinstance(second, Particle):
            sphere_first = first.get_position()
            sphere_second = second.get_position()
            radial_sep = sphere_second - sphere_first
            radial_distance = np.sqrt(np.dot(radial_sep, radial_sep))
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
        collision = False
        particle_position = particle_object.get_position()
        body_faces = body_object.get_faces()
        for index,face in enumerate(body_faces):
            on_plane = False
            face_versor = face.get_surface_vectors()
            face_position = face.get_position()
            distance_face_point = np.abs(np.dot(particle_position -
                              face_position, face_versor))
            #print(f"INDEX: {index}",distance_face_point, face_position )
            if distance_face_point <= self.collision_max_distance:
                on_plane = True
                print("HIT", face_position, particle_position)
            if on_plane:
                bool_array_dot_product = np.array([], dtype=bool)
                center_face_to_particle = particle_position - face_position
                vertex_per_face = face.get_vertex_position()
                for vertex in vertex_per_face:
                    vertex_to_particle = particle_position - vertex
                    if np.dot(center_face_to_particle, vertex_to_particle)>0:
                        dot_result = np.array([True], dtype = bool)
                    else:
                        dot_result = np.array([False], dtype = bool)
                    np.append(bool_array_dot_product, dot_result)
                if np.sum(np.array(bool_array_dot_product)) < len(vertex_per_face):
                    collision = True
        if collision:
            print("Inside", f"Point position: {particle_position}")


    def face_to_face_collision(self, first : Body, second : Body):
        first_versors = np.array([face.get_surface_vectors() for face in first.get_faces()])
        second_versors = np.array([second.get_surface_vectors() for face in second.get_faces()])
        print(first_versors)
        print(second_versors)

    def momentum_collision_heterogeneous(self, first : Body, second : Body or Particle, position : np.array):
        pass


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

    def force(self, y, t, mass):
        """ f = m * a -> f = m * dv/dt"""
        f = np.array([0,0,np.cos(y[2])])
        dv_dt = f/mass
        return dv_dt

    def posi(self, y, t):
        dr_dt = y
        return dr_dt

    def integrate(self, step):
        for obj in self.object_subscribed:
            t = np.linspace(start=self.time, stop=self.time + step, num=101)
            vel0 = obj.get_velocity()
            #pos = obj.get_position()
            mass = obj.get_mass()
            #r = odeint(func=self.posi, y0 = pos, t=t)
            v = odeint(func = self.force, y0 = vel0, t=t, args=(mass,))
            obj.set_velocity(v[-1])
            #obj.set_position(r[-1])

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
    """
    particles = [Particle((i+1),
                         [np.random.uniform(0,0),
                          np.random.uniform(0,0),
                          np.random.uniform(-.5,.5)],
                         [np.random.uniform(0,0),
                          np.random.uniform(0,0),
                          np.random.uniform(-0,0)]
                          )
                 for i in np.arange(dim)]
    particles.append(Cube(1,1,1,[0,0,3],[0,0,0],[0,0,0],[0,0,0], False))
    """
    particles = [Particle(1,[0.8,0.8,1],[0,0,0]),
                 Cilinder(1,1,8,10,0,[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0., 0.,1.], False)]
    space_instance.subscribe_objects_into_space(particles)
    step = 0.01
    condition = True
    time_start = 0.0
    counter = 0
    collision = 0
    total_energy = np.sum(np.array([obj.kinetic_energy() for obj in space_instance.get_objects_in_space()]))
    print(np.array([obj.kinetic_energy() for obj in space_instance.get_objects_in_space()]))
    quit()
    while condition:
        time_start += step
        collision = space_instance.update(step)
        print(space_instance.object_subscribed[1].faces[0].get_position())
