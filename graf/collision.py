import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


with open("particles_coll.dat", "r") as file:
    data_raw = file.read()

data_clip = data_raw.split("\n")
data_list = [data.split(" ") for data in data_clip]
data_array = np.array(data_list[:-1])

time_array = data_array[:,0]
collision_array = data_array[:,1]

to_df = {"time" : time_array, "collision" : collision_array}

data_frame = pd.DataFrame(to_df)
data_frame["time"] = data_frame["time"].astype(np.float64)
data_frame["collision"] = data_frame["collision"].astype(np.float64)

#data_frame["bin"] = pd.cut(data_frame["time"], bins=[i for i in range(0,50,10)], include_lowest=True)
#print(data_frame)

fig, ax = plt.subplots()

data_frame["collision"].hist(bins=5)
#plt.savefig("bars.png")
plt.show()
