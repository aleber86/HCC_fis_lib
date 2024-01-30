import sys
from cube import Cube
from particle import Particle
from space import Space
from render import render_func
import numpy as np


def main():
    p_size = 1.e-2
    p_mass = 1
    c_mass = 3
    zeros = np.zeros((3,))
    p_1_position = np.array([-0.5, 5.0, 0.5])
    p_1_velocity = np.array([0.1,7,0.1])
    a_position = np.array([np.pi/2,0,np.pi/4])
    a_2_position = np.array([0,0,-np.pi/4])
    c_position = np.array([0,2,0])
    #Defines initial conditions of objects
    elements = np.array([
                Cube(2, p_mass,0, p_1_position,a_position, zeros, zeros, False),
                Cube(3, c_mass, 0, c_position, zeros, zeros, zeros, False)])
    space_instance = Space()
    #Suscribe elements into space instance.
    space_instance.subscribe_objects_into_space(elements)

    #Defines time step using velocity
    step = p_size / (np.linalg.norm(p_1_velocity) * 1.e+1)

    stop_time = 10
    number_of_iter = int(stop_time / step)
    object_in_space = space_instance.get_objects_in_space()
    for iteration_step in np.arange(number_of_iter):
        if iteration_step % 100 == 0:
            render_func(object_in_space, f"_top_{iteration_step}", rec = False)
            #render_func(object_in_space, f"_side_{iteration_step}", (25, -5), True)
        space_instance.update(step)
    sys.exit(0)


if __name__ == '__main__':
    main()





