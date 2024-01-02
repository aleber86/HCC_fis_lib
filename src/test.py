from body import Body
import numpy as np
from plane import Plane

class cild:
    def __init__(self):
        self.plane_side =0
        self.height = 1
        self.radius = 1
        self.faces = None
        self.number_of_lateral_faces = 4
    def init_faces(self, in_vel, in_rot, fric, des, mass = 0.):
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
        #
        self.faces = np.array([Plane([self.height,
                                     self.plane_side,0],
                                     mass,
                                     fric,
                                     self.radius*np.array(
                                         [np.cos(point*degrees),np.sin(point*degrees),0.]),
                                      [0.,-np.pi/2., degrees*point]
                                      ,in_vel, in_rot, des)

                              for point in np.arange(points_per_base)],dtype = Plane)
        #Definition of the top and bottom vertex
        vertex_top_bottom_face = np.sqrt(self.radius**2 + self.plane_side**2)* np.array([[np.cos(deg*degrees),
                                                         np.sin(deg*degrees),
                                                         0] for deg in np.arange(points_per_base)])
        #Bottom first then top faces. Uses the edges and vertex defined before
        print(vertex_top_bottom_face)
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
if __name__ == '__main__':
    c = cild()
    c.init_faces([0,0,0], [0,0,0], 0, False)
    print(c.faces)

