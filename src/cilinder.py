import numpy as np
from body import Body
from plane import Plane


class Cilinder(Body):
    def __init__(self, diameter, height, num_lat_faces, mass,
                 friction, init_pos, init_angular, init_vel,
                 init_rot, destructive, _wp = np.float64):
        """
        Creates 3D mesh of regular body, Cilinder and other regular bodys
        Construct the object using the number of faces and mesurements
        """
        self.change_variable_state = True
        self.radius = diameter/2
        self.height = height/2
        self.number_of_lateral_faces = num_lat_faces
        self._wp = _wp
        self.plane_side = 0.0
        self.faces = None
        self.__init_faces([0,0,0], [0,0,0], friction, destructive)
        vertex, edges_indexes = self.__vertex_to_object__index_edges()
        Body.__init__(self, mass, vertex, friction, init_pos,
                      init_angular, init_vel, init_rot,
                      destructive, edges_indexes)
        self.vector_surface()
        self.__inertia_tensor()
        #self.volume = self.volume_calc()

    def __inertia_tensor(self):
        diag_1_1 = 1/12 * self.mass * (3*self.radius**2 + self.height**2)
        diag_2_2 = diag_1_1
        diag_3_3 = 0.5 * self.mass * self.radius
        self.inertia_tensor = np.array([[diag_1_1,0,0], [0,diag_2_2,0], [0,0,diag_3_3]])
        self.inertia_tensor_inverse = np.linalg.inv(self.inertia_tensor)

    def __init_faces(self, in_vel, in_rot, fric, des, mass = 0.):
        """
        Defines the position of top-bottom and lateral faces of a regular body.
        It divides the azimut angle in equal parts by the number of faces.
        Sets self.plane_side variable that depends on the angle.
        """
        points_per_base = self.number_of_lateral_faces
        degrees = 2.0*np.pi  / points_per_base #Azimut angle
        self.plane_side = np.tan(degrees/2.)*self.radius #Length of lateral sides

        """
        Creates lateral faces. Defines the initial point of everyone and makes two rotations:
        [0., -pi/2, degrees*point], point is the step.
        Rotation on Y axis is fixed so the planes sides surface vectors points outside of
        the body. And Z axis rotation gives the angular position of every face to cover the body
        """
        self.faces = np.array([Plane([self.height, self.plane_side,0],
                                     mass,
                                     fric,
                                     self.radius*np.array(
                                         [np.cos(point*degrees),np.sin(point*degrees),0.]),
                                      [0.,-np.pi/2., degrees*point]
                                      , in_vel, in_rot, des)
                               for point in np.arange(points_per_base)]
                              ,dtype = Plane)

        #Definition of the top and bottom vertex
        vertex_top_bottom_face = np.sqrt(self.radius**2 + self.plane_side**2)* np.array([[np.cos(deg*degrees),
                                                         np.sin(deg*degrees),
                                                         0] for deg in np.arange(points_per_base)])
        #Bottom first then top faces. Uses the edges and vertex defined before
        top_bottom = np.array([Plane(2,
                                     mass,
                                     fric,
                                     [0,0,i*self.height],[(1+i)*np.pi/2.,0,degrees/2.],
                                     in_vel,
                                     in_rot,
                                     des,
                                     vertex_top_bottom_face)
                                     for i in [-1,1]], dtype=Plane)
        self.faces = np.append(self.faces, top_bottom, axis=0)


    def __vertex_to_object__index_edges(self ):
        """
        Adds vertex and edges from the faces to the Object (Cilinder or child)
        """
        vertex = self.faces[0].get_vertex_position()
        edges = self.faces[0].get_edges_index()
        for face  in self.faces[1:]:
            vertex_position = face.get_vertex_position()
            vertex = np.vstack((vertex, vertex_position))
            #len(edges) defines the offset of indexing
            edges = np.vstack((edges, face.get_edges_index()+len(edges)))
        return vertex, edges
    """
    def update_faces(self, time):
        offset = 0
        for index,face in enumerate(self.faces):
            vertex_len = len(face.get_vertex_position())
            self.faces[index].set_vertex_position(self.global_vertex[offset:offset+vertex_len])
            self.faces[index].vector_surface()
            offset += vertex_len
    """

    def get_surface_vectors(self):
        return self.surface_vector

    def vector_surface(self):
        """Surface vectors are reffered to global system. It will auto-rotate
           Body class method OVERLOADED
        """
        self.surface_vector = np.array([face.get_surface_vectors() for face in self.faces])

if __name__ == '__main__':
    cil = Cilinder(1,1,8,4,0,[0,0,0], [0,0,0], [0,0,0], [0,0,0], False)
    faces = cil.get_faces()
    vertex = np.array([face.get_vertex_position() for face in faces])
