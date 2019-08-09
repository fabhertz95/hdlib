"""
plot hdlib data
"""
import numpy as np
import matplotlib.pyplot as plt

CPU = np.genfromtxt("cpu_data.csv", delimiter=",")
GPU = np.genfromtxt("gpu_data.csv", delimiter=",")
GPU_BATCH = np.genfromtxt("gpu_batch_data_16.csv", delimiter=",")

plt.scatter(CPU[:, 0], CPU[:, 1])
plt.scatter(GPU[:, 0], GPU[:, 1])
plt.scatter(GPU_BATCH[:, 0], GPU_BATCH[:, 1])

plt.show()
