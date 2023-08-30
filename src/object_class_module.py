import numpy as np
from matplotlib import pyplot as plt


"""
CALCULAR LAS COLOSIONES CON METODO DE MONTE CARLO
LAS DISTANCIAS ENTRE LOS OBJETOS Y LAS DIRECCIONES
DE SUS CARAS SUPERFICIALES NOS DAN UNA ESTIMACION
DE LA POSICION DEL OBJETO EN EL ESPACIO
"""

class Particle:
    def __init__(self, mass : float, init_pos : list or np.array = [0., 0., 0.],
                 init_vel : list or np.array = [0., 0., 0.],
                 init_external_force : list or np.array = [0.,0.,0.] , friction = 0.8, _wp = np.float64):

        self.position = np.array(init_pos, dtype = _wp)
        self.velocity = np.array(init_vel, dtype= _wp)
        self.external_force_interaction = np.array(init_external_force, dtype = _wp)
        self.mass = _wp(mass)
        self.remove = False #Existence: {True : subscribe, False : unsuscribe}
        self.friction = friction

    def get_external_force(self):
        return self.external_force_interaction

    def get_mass(self):
        return self.mass

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity

    def get_state(self):
        return self.remove

    def get_friction(self):
        return self.friction

    def set_external_force(self, external):
        self.external_force_interaction = external

    def set_position(self, actual_position : list or np.array):
        self.position = actual_position

    def set_velocity(self, actual_velocity : list or np.array):
        self.velocity = actual_velocity

    def set_state(self, actual_state : bool):
        self.remove = actual_state


class Body:
    def __init__(self, mass : float, vertex, friction : float,
                  init_pos : list or np.array, init_vel : list or np.array,
                  init_rot : list or np.array, destructive : bool,
                  edges_indexes, _wp = np.float64):
        r"""Class of 3D objects (mesh only).
        mass : represents the mass of the body
        vertex : sets the position of every vertex that defines the bodys
        on local coordinates.
        [[x_1,y_1,z_1],...,[x_n, y_n, z_n]]
        friction : represnts bodys friction
        init_pos/init_vel/init_rot : initial phase space
        init_pos : [x,y,z,\phi,\theta,\Phi] (body center)
        init_vel : [\dot{x}, \dot{y}, \dot{z}]
        init_rot : [\dot{\phi}, \dot{\theta}, \dot{\Phi}]
        destructive : if True, it breaks on collision
        edges : defines 2d tuples for edges of the 3D object
        [[v_1, v_2], [v_1, v_3], ...., [v_i, v_k]]
        _wp : working precision
        """
        self._wp = _wp
        self.mass = _wp(mass)
        self.friction = _wp(friction)
        self.position = np.array(init_pos, dtype = _wp)
        self.linear_velocity = np.array(init_vel, dtype = _wp)
        self.rotation_velocity = np.array(init_rot, dtype = _wp)
        self.destructive = destructive
        self.edges_indexes = np.array(edges_indexes, dtype = np.int32)
        self.volume = 1.0 #Implemented by self.volume_calc funcion
        self.density = 1.0 #Implemented by self.density_calc function
        self.local_vertex = _wp(vertex)
        self.global_vertex = 0.0
        self.__vertex_position(0.0)
        self.edges = self.edges_change()
        self.inertia_tensor = np.zeros((3,3), dtype = _wp)
        self.axial_vectors_to_faces = None
        self.state = True
        self.surface_vector = None

    def volume_calc(self):
        """Function implemented by Monte Carlo simulation as in CAD systems
        to calculate volume of 3D bodys"""
        pass

    def density_calc(self):
        """Function sets the density of bodys. All bodys are treated as homogenous.
        rho = mass / volume
        """
        dens = self.mass/self.volume
        return dens
    def get_mass(self):
        return self.mass

    def get_edges(self):
        return self.edges

    def get_linear_velocity(self):
        return self.linear_velocity

    def get_position(self):
        return self.position

    def get_rotation_velocity(self):
        return self.rotation_velocity

    def get_vertex_position(self):
        return self.global_vertex

    def get_vertex_position_local(self):
        return self.local_vertex

    def get_edges_index(self):
        return self.edges_indexes

    def get_friction(self):
        return self.friction

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def set_axial_vector_to_faces(self, vectors_to_center : np.array):
        self.axial_vectors_to_faces = vectors_to_center

    def set_edges(self, edg):
        self.edges = edg

    def set_linear_velocity(self, velocity):
        self.linear_velocity = velocity

    def set_position(self, position):

        self.position = position

    def set_rotation_velocity(self, rotation):
        self.rotation_velocity = rotation

    def set_vertex_position(self, position_of_vertex : np.array):
        self.global_vertex = position_of_vertex
    """
    def __local_to_global_vertex(self):
        center_of_figure, angular_deviation = self.position
        global_vertex_coordinates_rot = np.apply_along_axis(self.__vector_rotation, 1, self.local_vertex,
                                                            angular_deviation, np.zeros(center_of_figure.shape))

        global_vertex_coordinates = np.apply_along_axis(lambda x : x+center_of_figure, 1, global_vertex_coordinates_rot)
        return global_vertex_coordinates
    """
    def __vector_rotation(self, vector_to_rotate, angular_deviation, origin):
        """Rotation matrix = R_z(alpha)*R_y(beta)*R_x(gamma) | RotMat * VECTOR"""

        gamma, beta, alpha = angular_deviation
        rotational_matrix = np.matrix([[np.cos(alpha)*np.cos(beta),
                                       np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma),
                                       np.cos(alpha)*np.sin(beta)*np.cos(gamma)+ np.sin(alpha)*np.sin(gamma)],
                                      [np.sin(alpha)*np.cos(beta),
                                       np.sin(alpha)*np.sin(gamma)+np.cos(alpha)*np.cos(gamma),
                                       np.sin(alpha)*np.sin(beta)*np.cos(gamma)-np.cos(alpha)*np.sin(gamma)],
                                      [-np.sin(beta), np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma)]],
                                      dtype = self._wp)
        diff = vector_to_rotate - origin
        row = diff.shape[0]
        diff_column = np.reshape(diff, (row, 1))
        vector_rotated = np.matmul(rotational_matrix, diff_column)
        return vector_rotated

    def __vertex_position(self, delta_time : float = 0.0, axis : np.array or None = None):
        """Updates center position and vertex position on global system
            If axis = None, the body rotates over an axis thus it center (not the mass center,
            body defined center by instant position)"""

        delta_time = self._wp(delta_time)
        linear_velocity = self.get_linear_velocity()
        rotational_velocity = self.get_rotation_velocity()
        center_position, angular_deviation = self.get_position()
        center_update_position = self.__change_by_time(center_position , linear_velocity, delta_time)
        angular_update_direction = self.__change_by_time(angular_deviation , rotational_velocity, delta_time)
        if axis is None:
            origin = np.zeros(center_position.shape)
        else:
            origin = axis

        vertex_update_position = np.apply_along_axis(self.__vector_rotation, 1, self.local_vertex,
                                                     angular_update_direction, origin )

        vertex_update_position = np.array(vertex_update_position+center_update_position)
        self.set_position(np.array([center_update_position, angular_update_direction], dtype = self._wp))
        self.set_vertex_position(vertex_update_position)

    def __change_by_time(self, change_argument, change_rate, delta_time):
        """Change on phase space"""
        changed_argument = change_argument + change_rate * delta_time
        return changed_argument


    def __define_edges(self, indexes):
        edge = np.array([self.global_vertex[indexes[0]], self.global_vertex[indexes[1]]], dtype = self._wp)
        return edge

    def edges_change(self):
        edges = np.apply_along_axis(self.__define_edges, 1, self.edges_indexes)
        return  edges

    def vector_surface(self):
        pass

    def update(self, delta_time : float):
        delta_time = self._wp(delta_time)
        self.__vertex_position(delta_time)
        #self.define_axial_vectors()
        edge = self.edges_change()
        self.set_edges(edge)
        self.vector_surface()



class Plane(Body):
    """Primitive mesh body. Child of Body class."""
    def __init__(self, size, mass, friction, init_pos, init_vel, init_rot, destructive, _wp = np.float64):
        self.size = size
        vertex = np.array([[1.,1.,0.], [-1.,1.,0.], [-1.,-1.,0.], [1.,-1.,0]], dtype = _wp) * size/2.0
        edges_indexes  = np.array([(0,1), (1,2), (2,3), (3,0)], dtype = np.int32)
        Body.__init__(self, mass, vertex, friction, init_pos, init_vel, init_rot, destructive, edges_indexes)
        self.vector_surface()

    def vector_surface(self):
        """Surface vectors are reffered to global system. It will auto-rotate
           Body class method OVERLOADED
        """
        vertex_on_global = self.get_vertex_position()
        side_a = vertex_on_global[self.edges_indexes[0][1]] - vertex_on_global[self.edges_indexes[0][0]]
        side_b = vertex_on_global[self.edges_indexes[1][1]] - vertex_on_global[self.edges_indexes[1][0]]
        self.surface_vector = np.cross(side_a, side_b)


    def get_surface_vectors(self):
        return self.surface_vector

class Cube(Body):
    """Primitive mesh body. Child of Body class."""
    def __init__(self, side_size, mass, friction, init_pos, init_vel, init_rot, destructive, _wp = np.float64):
        self.side_size = side_size
        axial_vectors = self.__init_faces_position(side_size, _wp)
        self.faces = np.array([
            Plane(side_size, mass, friction, axial_vector_position, init_vel, init_rot, destructive)
            for axial_vector_position in axial_vectors], dtype=Plane)
        vertex, edges_indexes = self.__vertex_to_object__index_edges(init_pos)
        Body.__init__(self, mass, vertex, friction, init_pos, init_vel, init_rot, destructive, edges_indexes)

    def update_cube_faces(self, time):
        np.array([face.update(time) for face in self.faces])


    def get_surface_vectors(self):

        return self.surface_vector

    def vector_surface(self):
        """Surface vectors are reffered to global system. It will auto-rotate
           Body class method OVERLOADED
        """
        self.surface_vector = np.array([face.get_surface_vectors() for face in self.faces])

    def __init_faces_position(self,side_size,  _wp):
        """Defines the position of faces (instances of Plane)"""
        position_of_faces = side_size/2.*np.array([[1.,0.,0],
                                                   [0.,0.,1.],
                                                   [-1.,0.,0.],
                                                   [0.,0.,-1.],
                                                   [0.,1.,0.],
                                                   [0.,-1.,0.]], dtype=_wp)

        angular_position_of_faces = np.array([[0.,np.pi/2.,0.],
                                              [0.,0.,0.],
                                              [0.,-np.pi/2.,0.],
                                              [np.pi,0.,0.],
                                              [np.pi/2.,0.,0.],
                                              [-np.pi/2.,0.,0.]], dtype=_wp)
        position = np.array([[i,j] for i,j in zip(position_of_faces, angular_position_of_faces)])
        return position

    def __vertex_to_object__index_edges(self, position_to_cube_center):
        center, _ = position_to_cube_center
        vertex = self.faces[0].get_vertex_position() - center
        edges = self.faces[0].get_edges_index()
        counter = 1
        for face  in self.faces[1:]:
            vertex_position = face.get_vertex_position() - center
            vertex = np.vstack((vertex, vertex_position))
            edges = np.vstack((edges, face.get_edges_index()+counter*4))
            counter+=1
        return vertex, edges


def render(objects_array):
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
        for point in vertex_array:
            x,y,z = point
            ax.scatter(x,y,z, c='blue', marker='o', s=10)
        for edge in edge_array:
            x = [edge[0][0],edge[1][0]]
            y = [edge[0][1],edge[1][1]]
            z = [edge[0][2],edge[1][2]]
            ax.scatter(x,y,z, c='black', s=5)
            ax.plot(x,y,z, color='black')
    ax.scatter([0,1], [0,0], [0,0], c='blue', s=5)
    ax.plot([0,1], [0,0], [0,0], c='blue')
    ax.scatter([0,0], [0,1], [0,0], c='red', s=5)
    ax.plot([0,0], [0,1], [0,0], c='red')
    ax.scatter([0,0], [0,0], [0,1], c='green', s=5)
    ax.plot([0,0], [0,0], [0,1], c='green')
    plt.show()

if __name__ == '__main__':
    """
    cube_vertex = np.array([[1., 1., 1.], [-1.,1.,1.],
                   [1., -1., 1.], [-1., -1., 1.], [1.,-1.,-1.],
                   [1.,1.,-1.], [-1.,1.,-1.], [-1., -1., -1.]], dtype = np.float64)
    edges = np.array([(0,1),
                      (0,2),
                      (1,3),
                      (2,3),
                      (0,5),
                      (1,6),
                      (2,4),
                      (3,7),
                      (5,4),
                      (5,6),
                      (4,7),
                      (6,7)], dtype = np.int32)
    """

    cube = Cube(5.0, 1.0, 0.0, [[0.,0.,2.], [0.,0.,0.]], [0.,0.,0.,], [0.,10.,10], False )
    plane = Plane(8., 1., 0., [[0,0,0], [0,1,0]], [0,0,0], [0,0,0], False )
    objects = np.array([cube, plane])
    for _ in range(100):
        #print(f"center position / angles: {cube.get_position()} \n")
        #print(f"vertex_pos : {cube.get_vertex_position()} \n")
        #print(f"edge_position : {cube.get_edges()} \n")
        #print(f"LOCAL VERTEX: \n {cube.local_vertex} \n")
        #print(f"GLOBAL VERTEX: \n {cube.get_vertex_position()}\n")
        anim = render(objects)
        for obj in objects:
            obj.update(0.1)
            if isinstance(obj, Cube):
                obj.update_cube_faces(0.1)
                #print(obj.get_surface_vectors())
        #render(plane.get_vertex_position(), plane.get_edges())
        #plane.update(0.1)
