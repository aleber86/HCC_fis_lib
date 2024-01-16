import numpy as np
from body import Body


class Plane(Body):
    """Primitive mesh body. Child of Body class."""
    def __init__(self, size, mass,  friction, init_pos,
                 init_angular, init_vel, init_rot, destructive,
                 vertex_input = None,
                 edges_input  = None,_wp = np.float64):
        self.change_variable_state_position = True
        self.change_variable_state_angular = True
        self.change_variable_state_rotation_v = True
        self.change_variable_state_vel = True
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
        self.sides_local= None
        self.vector_surface()

    def volume_calc(self):
        pass

    def update_faces(self):
        pass

    def surface_calc(self, quant = 1000, n = 20):
        surface_array = []
        for _ in np.arange(n):
            counter = [0]
            vertex_global = self.global_vertex
            max_x, min_x = np.max(vertex_global[:,0]), np.min(vertex_global[:,0])
            max_y, min_y = np.max(vertex_global[:,1]), np.min(vertex_global[:,1])
            max_z, min_z = np.max(vertex_global[:,2]), np.min(vertex_global[:,2])
            surface_encaps = np.abs(max_x - min_x)*np.abs(max_y - min_y)
            point_array_random = np.array([[np.random.uniform(min_x,max_x),
                                   np.random.uniform(min_y,max_y),
                                   np.random.uniform(min_z, max_z)] for _ in np.arange(quant)], dtype=self._wp)
            np.apply_along_axis(self._inside_of_plane, 1, point_array_random,
                                             vertex_global, counter)
            surface_numeric = surface_encaps * counter[0] / quant
            surface_array.append(surface_numeric)
        mean_surface_value = np.mean(surface_array)
        return mean_surface_value



    def _inside_of_plane(self, point, array_vertex, counter):
        center_to_point = point + self.position
        for vertex in array_vertex:
            bool_array = []
            vertex_to_point = point - vertex
            if np.dot(center_to_point,vertex_to_point)<0.:
                bool_array.append(True)
            else:
                bool_array.append(False)


        if np.sum(bool_array)!=len(array_vertex):
            counter[0] = counter[0] + 1


    def vector_surface(self):
        """Surface vectors (versor) are reffered to global system. It will auto-rotate
           Body class method OVERLOADED.
           Sets sides of the plane
        """
        vertex_on_global = self.get_vertex_position() #Vertex on global system
        self.sides = np.array([vertex_on_global[self.edges_indexes[index][1]]
                               - vertex_on_global[self.edges_indexes[index][0]]
                               for index, _ in enumerate(self.edges_indexes)])

        vector_cross_product = np.cross(self.sides[0], self.sides[1])
        self.surface_vector = vector_cross_product / np.sqrt(np.dot(vector_cross_product, vector_cross_product))
        self.surface_vector = self.surface_vector + self.position
    def get_surface_vectors(self):
        return self.surface_vector


if __name__ == '__main__':
    plane_test = Plane(10, 0, 0, [1,10,1], [0,1,1], [0,0,0], [0,0,0], False)
    surface = plane_test.surface_calc()
    print(surface)
