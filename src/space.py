import numpy as np
from object_class_module import render
from body import Body
from particle import Particle
from cube import Cube
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
                object_x.set_velocity(-vel)
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

    def collision_detect(self, first, second, radius = 1.e-5):
        collision = False
        if isinstance(first, Particle) and isinstance(second, Particle):
            sphere_first = first.get_position()
            sphere_second = second.get_position()
            radial_sep = sphere_second - sphere_first
            radial_distance = np.sqrt(np.dot(radial_sep, radial_sep))
            if radial_distance <= radius:
#                print("PARTICLE-PARTICLE COLLISION")
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
            if self.__overlap_box(body_instance, particle_instance):
                #print("BODY - PARTICLE COLLISION")
                self.point_to_face_collision(body_instance, particle_instance)

                collision = True
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
        particle_position = particle_object.get_position()
        body_faces = body_object.get_faces()
        body_face_versors = np.array([face.get_surface_vectors() for face in body_faces])
        body_vertex = np.array([face.get_vertex_position() for face in body_faces])
        body_face_position = np.array([face.get_position() for face in body_faces])

        for index,center,versor in enumerate(zip(body_face_position, body_face_versors)):
            on_plane = False
            if np.abs(np.dot(particle_position - center, versor)) <= 1.e-5:
                on_plane = True
                print("IMPACT", center, particle_position)
            if on_plane:
                for vertex in body_vertex[index]:
                    pass




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

    def update(self, time):
        """Updates time, position and velocity for every object using odeint from Scipy. Time span is
        calculated with numpy linspace 10 elements, odeint calculates the interval and takes the last element
        to set new state for every object subscribed. If an element state is False then it will be unsuscribed."""

        collision = np.array([self.collision_detect(obj, sec)
                      for index,obj in enumerate(self.object_subscribed)
                      for sec in self.object_subscribed[index:]
                      if obj!= sec])

        np.array([obj.update(time) for obj in self.object_subscribed])
        np.array([obj.set_position(obj.get_position()+time*obj.get_velocity()) for obj in self.object_subscribed])
        self.box_limit()
        self.time = time
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
    particles = [Particle(1,[0.1,0.2,0],[0,0,0.1]),
                 Cube(1,1,0,[0,0,2],[0,0,0],[0,0,0],[0,0,0], False)]
    space_instance.subscribe_objects_into_space(particles)
    step = 0.01
    condition = True
    time_start = 0.0
    counter = 0
    collision = 0
    total_energy = np.sum(np.array([obj.kinetic_energy() for obj in space_instance.get_objects_in_space()]))
    while condition:
        time_start += step
        #render(particles, 1)
        collision = space_instance.update(step)
        """
        if collision>0:
            k=np.sum(np.array([obj.kinetic_energy() for obj in space_instance.get_objects_in_space()]))
            print(f"Step: {counter}, total time: {time_start}, Collisions: {collision} ")
            print(f"Total Energy: {total_energy} *** Actual Energy State: {k}")
            counter+=1
        """
