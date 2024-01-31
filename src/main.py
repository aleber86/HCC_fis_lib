import sys
from cube import Cube
from particle import Particle
from space import Space
from render import render_func
import numpy as np

def experim_1(sampling : int = 100, render_bool : bool = False, record = False ):
    np.random.seed(14568798)
    quant = 20
    elements = np.array([Particle(np.random.uniform(1,3), #size
                                   np.random.uniform(1,20),#mass
                                   [np.random.uniform(-2,2), #position
                                    np.random.uniform(-2,2),
                                    np.random.uniform(-2,2)],
                                   [np.random.uniform(-1,1), #velocity
                                    np.random.uniform(-1,1),
                                    np.random.uniform(-1,1)]
                                    ) for _ in np.arange(quant)])
    space_instance = Space()
    space_instance.subscribe_objects_into_space(elements)

    #Defines time step using velocity
    max_velocity = 7
    step = 1./max_velocity*1.e-2
    #step = p_size / (np.linalg.norm(p_1_velocity) * 1.e+1)

    stop_time = 50
    number_of_iter = int(stop_time / step)
    object_in_space = space_instance.get_objects_in_space()
    counter_collisions = 0
    time = 0
    collision_time = np.zeros((number_of_iter,2))
    energy_array = np.zeros((number_of_iter,quant+3))
    initial_en =np.sum(np.array([obj.kinetic_energy() for obj in object_in_space]))

    for iteration_step in np.arange(number_of_iter):
        energy = np.array([obj.kinetic_energy() for obj in object_in_space])
        energy_array[iteration_step][0] = time
        energy_array[iteration_step,1:-2] = energy
        total_en = np.sum(energy)
        energy_array[iteration_step,-2] = total_en
        energy_array[iteration_step,-1] = np.fabs(total_en - initial_en)

        if iteration_step % sampling == 0 and render_bool:
            print(iteration_step, number_of_iter)
            render_func(object_in_space, f"_side_particles_{iteration_step}",
                        rec = record, proyection=(25,15), p_size=20)

        counter_collisions = space_instance.update(step)
        time += step
        collision_time[iteration_step] = time,counter_collisions

    with open("particles_energy.dat", "w") as en_file:
        np.savetxt(en_file, energy_array)

    with open("particles_coll.dat", "w") as col_file:
        np.savetxt(col_file, collision_time)

def experim_2(sampling = 100, render_bool = False, record = False):
    #Iinital cond. cube_1
    velocity_1 = np.array([0,0,0])
    position_1 = np.array([1,1,1])
    angular_1 = np.array([0,0,0])
    rotation_1 = np.array([0,0,0])
    size_1 = 3
    mass_1 = 30
    cube_1 = Cube(size_1, mass_1, 0, position_1, angular_1, velocity_1, rotation_1, False)

    #Iinital cond. cube_2
    velocity_2 = np.array([0,1,0])
    position_2 = np.array([0.8,-3.,-0.8])
    angular_2 = np.array([np.pi/4,0,np.pi/4])
    rotation_2 = np.array([0,3,3])
    size_2 = 1
    mass_2 = 3
    cube_2 = Cube(size_2, mass_2, 0, position_2, angular_2, velocity_2, rotation_2, False)

    objects = [cube_1, cube_2]
    space_instance = Space()
    #Track the objects
    space_instance.subscribe_objects_into_space(objects)
    #Set time steps and iterations
    #time_step = 1 / np.linalg.norm(velocity_2) * 1.e-3
    time_step = 1.e-4
    time_stop = 0.5
    number_of_iter = int(time_stop/time_step)
    #Make sure all objects are tracked
    object_in_space = space_instance.get_objects_in_space()

    l_momentum_0 = np.array([obj.linear_momentum() for obj in object_in_space])
    a_momentum_0 = np.array([obj.angular_momentum() for obj in object_in_space])
    energy_0 = np.array([obj.kinetic_energy() for obj in object_in_space])

    total_l_momentum_0 = np.sum(l_momentum_0, 0)
    total_a_momentum_0 = np.sum(a_momentum_0, 0)
    total_energy_0 = np.sum(energy_0)

    array_to_save = np.zeros((number_of_iter, 21))

    time = 0
    for iteration_step in np.arange(number_of_iter):
        l_momentum = np.array([obj.linear_momentum() for obj in object_in_space])
        a_momentum = np.array([obj.angular_momentum() for obj in object_in_space])
        energy = np.array([obj.kinetic_energy() for obj in object_in_space])
        """
        array_to_save[iteration_step,0] = time
        array_to_save[iteration_step,1:4] = l_momentum[0]
        array_to_save[iteration_step,4:7] = l_momentum[1]
        array_to_save[iteration_step,7:10] = a_momentum[0]
        array_to_save[iteration_step,10:13] = a_momentum[1]
        array_to_save[iteration_step,13:15] = energy
        """
        total_l_m = np.sum(l_momentum, 0)
        total_a_m = np.sum(a_momentum, 0)
        total_energy = np.sum(energy)
        """
        array_to_save[iteration_step,15:18] = total_l_m
        array_to_save[iteration_step,18:21] = total_a_m
        array_to_save[iteration_step,-1] = total_energy
        """
        space_instance.update(time)
        if iteration_step%sampling == 0:
           # string_iter_total = str(number_of_iter)
           # string_iter = str(iteration_step)
           # long = len(string_iter_total)-len(string_iter)
           # save_name = long*'0'+string_iter
           # render_func(object_in_space,f"cube_side_{save_name}" ,rec=True, proyection=(15,3))
            print(total_energy, total_energy_0)

        time += time_step
    """
    with open("3D_sim.dat", "w") as file:
        np.savetxt(file, array_to_save)
    """



def main():
    #experim_1()
    experim_2(sampling = 50)
    sys.exit(0)


if __name__ == '__main__':
    main()





