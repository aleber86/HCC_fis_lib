import matplotlib.pyplot as plt
import numpy as np

ax = plt.figure().add_subplot(projection='3d')

#    plt.rcParams["figure.figsize"] = [20, 20]
#    plt.rcParams["figure.autolayout"] = True

#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection = '3d')
    #ax.legend()
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# Make the grid
#x, y, z = np.meshgrid(np.arange(0, 1, 1),
#                      np.arange(0, 1, 1),
#                      np.arange(0, 1, 1))
points = [[-3,3,0],[3,3,0],[-3,-3,0],[3,-3,0]]
for point in points:
    x,y,z = point
    ax.scatter(x,y,z, color="g")

ax.scatter(2,1,2, color="k", marker = "v")
ax.scatter(2.1,1.1,2.1, color="r", marker = "o")
ax.scatter(0,0,1, color="k", marker = "^")
ax.plot([points[0][0],points[2][0]], [points[0][1], points[2][1]], [points[0][2], points[2][2]], color="k")
ax.plot([points[1][0],points[3][0]], [points[1][1], points[3][1]], [points[1][2], points[3][2]], color="k")
ax.plot([points[2][0],points[3][0]], [points[2][1], points[3][1]], [points[2][2], points[3][2]], color="k")
ax.plot([points[0][0],points[1][0]], [points[0][1], points[1][1]], [points[0][2], points[1][2]], color="k")

ax.plot([0,0], [0,0], [0,1], color="k")
ax.plot([0,2], [0,1], [0,2], color="k" )
ax.plot([2.1, 2.1],[1.1, 1.1],[0,2.1], color="b", linestyle="-")
plt.show()


#0,0,0 -> 1,0,0
#1,0,0 -> 1,1,0
