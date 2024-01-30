import numpy as np
from particle import Particle
from body import Body
from matplotlib import pyplot as plt


def render_func(objects_array : np.array,
                name : str,
                proyection : tuple = (90,0),
                rec : bool = False,
                versors = False,
                hit_b = False,
                p_size = 5
                ) -> None:

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
    ax.view_init(*proyection)
    for obj in objects_array:
        if isinstance(obj, Body):
            vertex_array = obj.get_vertex_position()
            edge_array = obj.get_edges()
            for point in vertex_array:
                x,y,z = point
                ax.scatter(x,y,z, c='blue', marker='o', s=10)
            for edge in edge_array:
                x = [edge[0][0],edge[1][0]]
                y = [edge[0][1],edge[1][1]]
                z = [edge[0][2],edge[1][2]]
                ax.scatter(x,y,z, c='black', s=5)
                ax.plot(x,y,z, color='black')

            #Hit box plot
            if hit_b:
                hit_box = obj.collision_box()
                for point in hit_box:
                    x,y,z = point
                    ax.scatter(x,y,z, c='red', marker='o', s=10)
            if versors:
                for face in obj.get_faces():
                    pos = face.get_position()
                    vec = face.get_surface_vectors()
                    x = [vec[0],pos[0]]
                    y = [vec[1],pos[1]]
                    z = [vec[2], pos[2]]
                    ax.scatter(vec[0],vec[1],vec[2],marker='v',color="r")
                    ax.plot(x,y,z, c='k', marker='o')
        elif isinstance(obj, Particle):
            x,y,z = obj.get_position()
            ax.scatter(x,y,z, c='b', marker='x', s=p_size)
#            hit_box = obj.collision_box()
#            for point in hit_box:
#                x,y,z = point
#                ax.scatter(x,y,z, c='red', marker='o', s=10)

    ax.scatter([0,1], [0,0], [0,0], c='blue', s=5, marker = 'v')
    ax.plot([0,1], [0,0], [0,0], c='blue')
    ax.scatter([0,0], [0,1], [0,0], c='red', s=5, marker='v')
    ax.plot([0,0], [0,1], [0,0], c='red')
    ax.scatter([0,0], [0,0], [0,1], c='green', s=5, marker='v')
    ax.plot([0,0], [0,0], [0,1], c='green')
    if rec:
        plt.savefig(f"tmp/image{name}.png")
        plt.close()
    else:
        plt.show()

