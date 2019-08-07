"""
plot hdlib data
"""
import numpy as np
import matplotlib.pyplot as plt

CPU = np.genfromtxt("cpu_data_1000.csv", delimiter=",")
GPU = np.genfromtxt("gpu_data_1000.csv", delimiter=",")

print(CPU.shape)

plt.scatter(CPU[:,0], CPU[:,1])
plt.scatter(GPU[:,0], GPU[:,1])
plt.show()
