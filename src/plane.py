import numpy as np
from body import Body


class Plane(Body):
    """Primitive mesh body. Child of Body class."""
    def __init__(self, size, mass,  friction, init_pos,
                 init_angular, init_vel, init_rot, destructive,
                 vertex_input = None,
                 edges_input  = None,_wp = np.float64):

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
    """
    def ray_trace(self, other_object_face = None, quant = 300):
        center, _ = self.get_position()
        vertex_global = self.get_vertex_position()
        max_x, max_y, max_z = np.max(vertex_global, 0)
        min_x, min_y, min_z = np.min(vertex_global, 0)
        if np.isclose(np.abs(max_x-min_x),1.e-5):
            flatter = np.array([0,1,1])
            ind_dim = 0
        elif np.isclose(np.abs(max_y-min_y),1.e-5):
            flatter = np.array([1,0,1])
            ind_dim = 1
        else :
            flatter = np.array([1,1,0])
            ind_dim = 2
        vector_on_xy_plane = (center - vertex_global) * flatter
        normal_vector = self.get_surface_vectors()
        edges = self.get_edges() * flatter
        point_on_plane = []
        for times in np.arange(quant):
            point = np.array([np.random.uniform(min_x, max_x),
                              np.random.uniform(min_y, max_y),
                              np.random.uniform(min_z, max_z)])
            #vertex_flat = vertex_global * flatter
            position_from_vertex = (point - vertex_global)*flatter
            inside_value = np.array([True if np.dot(pos/np.linalg.norm(pos),vec)>=
                                     np.dot((edg[1]-edg[0])*flatter/np.linalg.norm((edg[1]-edg[0])*flatter), vec) else False
                                     for vec,pos,edg in zip(vector_on_xy_plane, position_from_vertex, edges)])
            if False in inside_value:
                pass
            else:
                point = self.__scalar_plane_function(ind_dim, normal_vector, point) + center
                point_on_plane.append(point )
        result_points = np.array(point_on_plane )
        return result_points

    def __scalar_plane_function(self, index_dimension, normal_vec, point):
        indexes = [0,1,2]
        indexes.pop(index_dimension)
        if not np.isclose(normal_vec[index_dimension],0.):
            point[index_dimension] = - (np.sum(np.array([normal_vec[ind]*point[ind] for ind in indexes]))
            /normal_vec[index_dimension])
        else:
            point[index_dimension] = 0.
        return point
    """
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
