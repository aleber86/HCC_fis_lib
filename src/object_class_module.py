import numpy as np
from matplotlib import pyplot as plt
from cube import Cube, Prisma
from plane import Plane
from cilinder import Cilinder
from sphere import Sphere




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
                        np.random.uniform(-10,10)], False) for _ in np.arange(5)])
    #plane = Plane(1, 1, 0, [[-10,0,0], [0,0,0]], [0,0,1], [0,0,1], False)
    #plane = Sphere(1.0,4,1,1,0, [[0,0,0], [0,0,0]], [2,0,0], [0,2,2], False)
    #plane = Cube(1,1,0,[[ -10, 6, 0],[3,4,1]], [1,1,1], [-3,-2,-1], False)
    for i in range(100):
        #print(objects[0].get_surface_vectors())
        render(objects, i)
        print(f"Frame {i+1} Calculated")
        objects[0].update(0.1)
            #print(obj.get_surface_vectors())
