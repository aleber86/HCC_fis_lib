import numpy as np
from matplotlib import pyplot as plt
from body import Body
from cube import Cube, Prisma
from plane import Plane
from cilinder import Cilinder
from particle import Particle




def render(objects_array, name):
    plt.rcParams["figure.figsize"] = [20, 20]
    plt.rcParams["figure.autolayout"] = True

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    #ax.legend()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    for obj in objects_array:
        if isinstance(obj, Body):
            vertex_array = obj.get_vertex_position()
            edge_array = obj.get_edges()
            hit_box = obj.collision_box()
            for point in vertex_array:
                x,y,z = point
                ax.scatter(x,y,z, c='blue', marker='o', s=10)
            for edge in edge_array:
                x = [edge[0][0],edge[1][0]]
                y = [edge[0][1],edge[1][1]]
                z = [edge[0][2],edge[1][2]]
                ax.scatter(x,y,z, c='black', s=5)
                ax.plot(x,y,z, color='black')
            for point in hit_box:
                x,y,z = point
                ax.scatter(x,y,z, c='red', marker='o', s=10)
        elif isinstance(obj, Particle):
            x,y,z = obj.get_position()
            ax.scatter(x,y,z, c='b', marker='x', s=5)
            hit_box = obj.collision_box()
            for point in hit_box:
                x,y,z = point
                ax.scatter(x,y,z, c='red', marker='o', s=10)

    ax.scatter([0,1], [0,0], [0,0], c='blue', s=5)
    ax.plot([0,1], [0,0], [0,0], c='blue')
    ax.scatter([0,0], [0,1], [0,0], c='red', s=5)
    ax.plot([0,0], [0,1], [0,0], c='red')
    ax.scatter([0,0], [0,0], [0,1], c='green', s=5)
    ax.plot([0,0], [0,0], [0,1], c='green')
    plt.show()
#    plt.savefig(f"tmp/image{name}.png")

if __name__ == '__main__':
    import time
    box_limit = 0.5*np.array([1,1,1])
    np.random.seed(1235048800)
    objects = np.array([Particle(np.random.uniform(0.1,5),
                                 [np.random.uniform(-0.1,0.1),
                                  np.random.uniform(-0.1,0.1),
                                  np.random.uniform(-00.1,0.01)],
                                 [np.random.uniform(-10,10),
                                  np.random.uniform(-10,10),
                                  np.random.uniform(-10,10)]) for _ in np.arange(70)])
#    objects = np.array([Cilinder(1,3,8,1,0,[0,0,0], [0,0,0], [0,0,0], [0,0,0], destructive=False)])
    start_time = time.perf_counter()
    counter = 1
#    with open("data_set.dat", "w") as file_out:
    for i in range(1000):
        if time.perf_counter() - start_time >=1:
            print(f"Frames / sec {counter} Calculated")
            start_time = time.perf_counter()
            counter = 0
        else:
            counter +=1
        coll = np.array([obj.collision_detect(oth) for obj in objects for oth in objects if obj != oth])
        tot_coll = np.sum(coll)
        if tot_coll != 0: print(tot_coll)

        if i%10 == 0:
            pass
            #render(objects, i)
            """
            arr = np.matrix(objects[0].get_position())
            for obj in objects:
                arr = np.vstack((arr,obj.get_position()))
            np.savetxt(file_out, arr)
            file_out.write(2*'\n')
            """
        np.array([obj.update(0.1) for obj in objects])
        #print(objects[0].get_angular_position())
