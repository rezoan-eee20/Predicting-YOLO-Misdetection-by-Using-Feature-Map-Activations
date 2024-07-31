import matplotlib.cm as cm
import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('bmh')

dataframe = pd.read_csv("neighbors_2.csv")
print(dataframe.head(10))
print(list(dataframe.columns))
df = pd.DataFrame(dataframe)
cols = [7]
df = df[df.columns[cols]]
# print(df.head())
x = dataframe['Detected Prob']
# plt.plot(range(len(x)), x)
# plt.show()


dataframe2 = pd.read_csv('neighbors_small.csv')
df2 = pd.DataFrame(dataframe2)
df2 = df2[df2.columns[cols]]

arr = dataframe2.to_numpy()
plt.figure()
arrays = []
stds = []
for i in range(int(len(arr)/9)):
    my_arr = arr[i*9:(i*9)+9, 7]
    my_arr_names = arr[i*9:(i*9)+9, 6][0]
    arrays.append(my_arr)
    stds.append(np.var(my_arr))
    misStr = "Correct" if i < 2 else "Misclassified"
    plt.plot(range(9), my_arr, label="Image 3: " +
             my_arr_names + " ({})".format(misStr))

plt.plot(range(9), x, 'r', label="Image 2: horse (Misclassified)")
plt.xlabel("Grid")
plt.ylabel("Class Scores")
plt.legend()
plt.show()

# plt.figure()
# plt.scatter(range(len(stds)), stds)
# plt.show()
# print(arrays)


# X, Y = np.meshgrid(range(len(arrays)), range(9))
# Z = arrays
# fig, ax = plt.subplots()
# CS = ax.contour(X, Y, Z)
# ax.clabel(CS, inline=1, fontsize=10)
# ax.set_title('Simplest default with labels')
# plt.show()


# m = dataframe2['Detected Prob']


# plt.figure()
# plt.plot(range(len(m)), m)
# plt.show()


# #COntour plot example
# delta = 0.025
# y = list(range(len(x)))

# X, Y = np.meshgrid(x, y)

# # Z1 = np.exp(-X**2 - Y**2)
# # Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
# # Z = (Z1 - Z2) * 2
# Z = range(len(x))

# fig, ax = plt.subplots()
# CS = ax.contour(X, Y, Z)
# ax.clabel(CS, inline=1, fontsize=10)
# ax.set_title('Simplest default with labels')

# plt.show()
# #Enf contoru plot
