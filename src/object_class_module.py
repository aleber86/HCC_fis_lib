import numpy as np
from matplotlib import pyplot as plt
from cube import Cube, Prisma
from plane import Plane
from cilinder import Cilinder




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
        vertex_array = obj.get_vertex_position()
        edge_array = obj.get_edges()
        hit_box = obj.collision_box()
        faces = obj.faces
#        scat = obj.array_point
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
        for face in faces:
            pos = [0,0,0]
            fpos, _ = face.get_position()
            x = [pos[0],fpos[0]]
            y = [pos[1],fpos[1]]
            z = [pos[2],fpos[2]]
            ax.scatter(x,y,z, c='green', s=5)
            ax.plot(x,y,z, color='green')
    """
        for point in scat:
            x,y,z = point
            ax.scatter(x,y,z, c='green', marker='o', s=10)
    """
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
    np.random.seed(123)
    """
    objects = np.array([Cilinder(np.random.randint(1,3),
                      np.random.randint(1,5),
                      np.random.randint(5, 10),
                      0.,
                      False,
                      [[np.random.uniform(-10,10),
                        np.random.uniform(-10,10),
                        np.random.uniform(-10,10)],
                        [np.random.uniform(-10,10),
                        np.random.uniform(-10,10),
                        np.random.uniform(-10,10)]],
                      [np.random.uniform(-10,10),
                        np.random.uniform(-10,10),
                        np.random.uniform(-10,10)],
                      [np.random.uniform(-10,10),
                        np.random.uniform(-10,10),
                        np.random.uniform(-10,10)], False) for _ in np.arange(2)])
    """

    objects = np.array([Cilinder(1,2,8,1,1,[[0,0,11*i], [0,0,0]],[0,0,0],[i,i,i], False) for i in np.arange(2)])
    #plane = Plane(1, 1, 0, [[-10,0,0], [0,0,0]], [0,0,1], [0,0,1], False)
    #plane = Cube(1,1,0,[[ -10, 6, 0],[3,4,1]], [1,1,1], [-3,-2,-1], False)
    start_time = time.perf_counter()
    counter = 1
    for i in range(1000):
        #print(objects[0].get_surface_vectors())
        #render(objects, i)
        if time.perf_counter() - start_time >=1:
            print(f"Frames / sec {counter} Calculated")
            start_time = time.perf_counter()
            counter = 0
        else:
            counter +=1
        #print(np.array([obj.get_surface_vectors() for obj in objects]))
        np.array([objects[i].collision_detect(objects[j])
                  for i in np.arange(len(objects)-1)
                  for j in np.arange(i+1, len(objects))] )
        np.array([obj.update(0.1) for obj in objects])
