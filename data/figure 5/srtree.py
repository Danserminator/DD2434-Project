import matplotlib.pyplot as plt
 
#************* 500 query points *****************
 
 
# figure 5
#  1-NNS
x = [1000, 2000, 5000, 10000, 19000,61265]
y = [6.53106, 11.2064, 19.1603, 31.998, 52.2124, 96.5531]
 
# 10-NNS
x = [1000, 2000, 5000, 10000, 19000,61265]
y = [8.28858, 14.2164, 24.7756, 41.7876, 68.6433, 133.78]
 
plt.xlabel('Number of database points')
plt.ylabel('Disk accesses')
plt.plot(x, y, linestyle='--', marker='o')
plt.show()
 
 
# figure 7
# 1-NN
x = [8, 16, 32]
y = [14.487, 37.3868, 94.4028]
 
# figure 7
# 10-NN
x = [8, 16, 32]
y = [17.481, 46.2485, 128.335]
 
plt.xlabel('Dimensions')
plt.ylabel('Disk accesses')
plt.plot(x, y, linestyle='--', marker='o')
plt.show()
 
 
#************* 1000 query points *****************
 
 
# figure 5
#  1-NNS
x = [1000, 2000, 5000, 10000, 19000,61265]
y = [5.72673, 10.8098, 20.0771, 33.1481, 46.2843, 89.5726]
 
# 10-NNS
x = [1000, 2000, 5000, 10000, 19000,61265]
y = [7.42543, 13.7027, 26.2823, 44.3193, 62.8308, 124.947]
 
plt.xlabel('Number of database points')
plt.ylabel('Disk accesses')
plt.plot(x, y, linestyle='--', marker='o')
plt.show()
 
 
# figure 7
# 1-NN
x = [8, 16, 32]
y = [13.4505, 35.6256, 88.1041]
 
# figure 7
# 10-NN
x = [8, 16, 32]
y = [14.8859, 43.8368, 120.718]
 
plt.xlabel('Dimensions')
plt.ylabel('Disk accesses')
plt.plot(x, y, linestyle='--', marker='o')
plt.show()
 
 
#*********** Comparison with 19k data set and different dimensions *****
# Just to try - 19k dataset
# figure 7
# 1-NN
# x = [8, 16, 32]
# y = [7.26052, 18.8236, 52.3347,]
 
# # figure 7
# # 10-NN
# x = [8, 16, 32]
# y = [8.51904, 23.1182, 70.5832]
 
# plt.xlabel('Dimensions')
# plt.ylabel('Disk accesses')
# plt.plot(x, y, linestyle='--', marker='o')
# plt.show()
