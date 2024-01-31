import matplotlib.pyplot as plt
import pandas as pd

with open("cilinder_surface_faces_volume_200.dat", "r") as file_read:
    data_raw = file_read.read()

data = data_raw.split("\n")
data = data[:-1]
data = data[1:]

points = []
volume_n = []
volume_a = []
err = []
for row in data:
    row_clean = row.split(' ')
    points.append(row_clean[0])
    volume_n.append(row_clean[1])
    volume_a.append(row_clean[2])
    err.append(row_clean[3])

dictionary = {"n_random_p" : points,
              "volume_numeric" : volume_n,
              "volume_analitic" : volume_a,
              "relative_err" : err}

data_frame = pd.DataFrame(dictionary)

data_frame.plot(kind="bar", x="n_random_p", y="relative_err")
plt.show()
#print(data_frame)

