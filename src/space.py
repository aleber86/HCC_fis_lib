import numpy as np
from object_class_module import Particle, Cube, Plane, render
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

    def get_object_telemetry(self):
        """Function returns time step and position/velocity of every subscribed object"""
        telemetry = np.array([])
        for obj in self.object_subscribed:
            if telemetry.size > 0:
                first = np.hstack((np.array([self.time]), np.append(obj.get_position(), obj.get_linear_velocity())))
                telemetry = np.vstack((telemetry,first))
            else:
                telemetry = np.hstack((np.array([self.time]), np.append(obj.get_position(), obj.get_linear_velocity())))

        return telemetry

    def update(self, time):
        """Updates time, position and velocity for every object using odeint from Scipy. Time span is
        calculated with numpy linspace 10 elements, odeint calculates the interval and takes the last element
        to set new state for every object subscribed. If an element state is False then it will be unsuscribed."""
        self.equation_of_mov(time)
        for obj in self.object_subscribed:
            obj.update(time)
        self.time = time

    def equation_of_mov(self, time):
        step_back_time = self.time
        t = np.linspace(step_back_time, time, 10)
        for index, obj in enumerate(self.object_subscribed):
            pos, ang = obj.get_position()
            vel = obj.get_linear_velocity()
            ext_force = 0.0
            argument = np.append(pos, vel)
            friction = obj.get_friction()
            resul = odeint(self.__interacion_in_space, argument, t, args = (ext_force,self.air, self.gravity, obj.get_mass()))
            if(resul[-1][2]<1.e-2):
                #self.object_subscribed[index].set_state(True)
                velocity_invertion = np.array([1.,1.,-friction], dtype=self._wp)
                position_invertion = np.array([1.,1.,0.], dtype=self._wp)
            else:
                velocity_invertion = np.ones((3,), dtype=self._wp)
                position_invertion = np.ones((3,), dtype = self._wp)
            #self.object_subscribed[index].set_position([resul[-1][:3].copy()*position_invertion,ang])
            self.object_subscribed[index].set_linear_velocity(resul[-1][3:].copy()*velocity_invertion+ext_force)

    def __interacion_in_space(self, pos_vel, y, ext_force, air, grav, mass):
        """
        Differential equation system.
        Degrees of freedom : 3
        Fluid resistance : Yes
        Gravity : Yes

        Movement of particles using a diff. eq. expressed by:

        \ddot{X} - K/m \dot{X}^2 - g = 0

        K : Air resistance coef.
        m : Mass of the particles
        g : Gravity

        """
        position = pos_vel[:3].copy()
        velocity = pos_vel[3:].copy()
        if position[2]>0.:
            dvelocity = - np.array([0.,0.,grav])
        else:
            dvelocity = np.zeros((3,))
            velocity = dvelocity

        dydt = np.append(velocity, dvelocity)

        return dydt

    def __str__(self):
        """Prints every attribute of Space class"""
        string = ""
        for item, values in self.__dict__.items():
            string = f"{string}{item} {values} {type(values)}\n"

        print(self.__dict__)
        return string

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    space_instance = Space()
    cube = Cube(1,1,0,[[0,0,20],[0,0,0]], [0,0,0], [0,0,0], False)
    plane = Plane(7, 1e6, 10, [[0,0,0],[0,0,0]], [0,0,0], [0,0,0], False)
    """
    particles = [Particle(np.random.randint(1,50),
                         8*np.ones((3,), dtype= np.float64),
                         2*np.ones((3,), dtype=np.float64))

                 for i in np.arange(10)]

    space_instance.subscribe_objects_into_space(particles)
    """
    space_instance.subscribe_objects_into_space(cube)
    space_instance.subscribe_objects_into_space(plane)
    to_print_z = np.array([])
    to_print_z_v = np.array([])
    for time_advance in np.arange(0,10):
        space_instance.update(0.1)
        render(space_instance.get_objects_in_space())
        telemetry = space_instance.get_object_telemetry()
        if telemetry.size>0:
            pass
