import numpy as np
from object_class_module import render
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
    def __init__(self, init_time = 0.0, gravity : float = 9.88,
                 air_res : float = 0.5, _wp = np.float64):

        self.gravity = _wp(gravity)
        self.object_subscribed = np.array([])
        self.time = init_time
        self.air = _wp(air_res)
        self.size_shepre_space = 1.e-1
        self._wp = _wp

    def subscribe_objects_into_space(self, object_to_subscribe):
        self.object_subscribed = np.append(self.object_subscribed, object_to_subscribe)

    def unsuscribe_objects(self, object_to_unsuscribe : list or tuple or np.ndarray or object = None):
        """Function removes elements subscribed to Space instance.
        It can recive iterable object with objects to unsuscribe"""
        if self.object_subscribed.size>0:
            res = np.apply_along_axis(lambda x : [el.get_state() for el in x] ,0, self.object_subscribed)
            self.object_subscribed = np.delete(self.object_subscribed, res)
        """
        if isinstance(object_to_unsuscribe, (list, tuple)):
            #If is an iterable cast -> np.ndarray
            object_to_unsuscribe = np.array(object_to_unsuscribe)

        if  (isinstance(object_to_unsuscribe, np.ndarray) and
        object_to_unsuscribe.shape == self.object_subscribed.shape):
            self.object_subscribed = np.delete(self.object_subscribed,
                                               np.where(self.object_subscribed==object_to_unsuscribe))
        elif(isinstance(object_to_unsuscribe, np.ndarray) and
             object_to_unsuscribe.shape != self.object_subscribed.shape):
            for el in object_to_unsuscribe:
                self.object_subscribed = np.delete(self.object_subscribed, np.where(self.object_subscribed == el))
        """

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
        if isinstance(first, Particle) and isinstance(second, Particle):
            sphere_first = first.get_position()
            sphere_second = second.get_position()
            radial_sep = sphere_second - sphere_first
            radial_distance = np.sqrt(np.dot(radial_sep, radial_sep))
            if radial_distance <= radius:
#                print("PARTICLE-PARTICLE COLLISION")
                self.momentum_collision_partice_particle(first, second)
                return True
            else:
                return False


        elif ((isinstance(first, Particle) and isinstance(second, Body)) |
              (isinstance(second, Particle) and isinstance(first, Body))):
            pass
        elif isinstance(first, Body) and isinstance(second, Body):
            pass


    def momentum_collision_partice_particle(self, first, second):
        """ELASTIC COLLISION"""
        first_velocity = first.get_velocity()
        second_velocity = second.get_velocity()
        first_mass = first.get_mass()
        second_mass = second.get_mass()

        first_new_velocity = (first_velocity*(first_mass-second_mass) +
                             2.*second_velocity*second_mass)/(first_mass+second_mass)

        second_new_velocity = (second_velocity*(second_mass-first_mass) +
                              2.*first_velocity*first_mass)/(first_mass + first_mass)

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

        np.array([obj.update() for obj in self.object_subscribed])
        np.array([obj.set_position(obj.get_position()+time*obj.get_velocity()) for obj in self.object_subscribed])
        self.box_limit()
        self.time = time
        total_collision = np.sum(collision)
        return total_collision

if __name__ == '__main__':
    np.random.seed(456791)
    dim = 1000
    space_instance = Space()
    particles = [Particle((i+1),
                         [np.random.uniform(-.005,.005),
                          np.random.uniform(-.005,.005),
                          np.random.uniform(-.005,.005)],
                         [np.random.uniform(-.01,.01),
                          np.random.uniform(-.01,.01),
                          np.random.uniform(-.01,.01)]
                          )
                 for i in np.arange(dim)]
    space_instance.subscribe_objects_into_space(particles)
    step = 0.1
    condition = True
    time_start = 0.0
    counter = 0
    total_energy = np.sum(np.array([obj.kinetic_energy() for obj in space_instance.get_objects_in_space()]))
    while condition:
        time_start += step
        collision = space_instance.update(step)
        if collision>0:
            k=np.sum(np.array([obj.kinetic_energy() for obj in space_instance.get_objects_in_space()]))
            print(f"Step: {counter}, total time: {time_start}, Collisions: {collision} ")
            print(f"Total Energy: {total_energy} *** Actual Energy State: {k}")
        counter+=1
