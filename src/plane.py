import numpy as np
from body import Body


class Plane(Body):
    """Primitive mesh body. Child of Body class."""
    def __init__(self, size, mass,  friction, init_pos,
                 init_angular, init_vel, init_rot, destructive,
                 vertex_input = None,
                 edges_input  = None,_wp = np.float64):
        self.change_variable_state = True
        if vertex_input is None:
            vertex = [[1.,1.,0.], [-1.,1.,0.], [-1.,-1.,0.], [1.,-1.,0]]
        else:
            vertex = vertex_input

        if edges_input is None:
            edges_ind = np.array([(i,i+1) for i in np.arange(len(vertex)-1)], dtype = np.int32)
            edges_ind = np.append(edges_ind, np.array([(len(vertex)-1,0)]), axis=0)
        else:
            edges_ind = edges_input

        if isinstance(size, (float, int)):
            self.size = _wp(size)
            vertex_array = np.array(vertex, dtype = _wp) * size/2.0
        elif isinstance(size, (list, np.array)):
            self.size = np.array(size, dtype=_wp)
            vertex = np.array(vertex, dtype=_wp)
            vertex_array = np.apply_along_axis(lambda x: self.size*x, 1, vertex)


        self.faces = None
        edges_indexes = np.array(edges_ind, dtype = np.int32)
        Body.__init__(self, mass, vertex_array, friction, init_pos,
                      init_angular, init_vel, init_rot, destructive, edges_indexes)
        self.sides = None
        self.vector_surface()
    def volume_calc(self):
        pass

    def update_faces(self):
        pass

    def vector_surface(self):
        """Surface vectors (versor) are reffered to global system. It will auto-rotate
           Body class method OVERLOADED.
           Sets sides of the plane
        """
        vertex_on_global = self.get_vertex_position() #Vertex on global system
        self.sides = np.array([vertex_on_global[self.edges_indexes[index][1]]
                               - vertex_on_global[self.edges_indexes[index][0]]
                               for index, _ in enumerate(self.edges_indexes)])

        vector_cross_product = np.cross(-self.sides[0], self.sides[1])
        self.surface_vector = vector_cross_product / np.sqrt(np.dot(vector_cross_product, vector_cross_product))

    def get_surface_vectors(self):
        return self.surface_vector

    def ray_trace(self, quant = 50):
        edges = self.get_edges()
        list_points = []
        for _ in np.arange(quant):
            point = (edges[0][1]-edges[0][0])
            list_points.append(point)
        result = np.array(list_points)
        return result

if __name__ == '__main__':
    plane_test = Plane(1, 0, 0, [[0,0,0], [0,0,0]], [0,0,0], [0,0,0], False)
    print(isinstance(plane_test, Body))
