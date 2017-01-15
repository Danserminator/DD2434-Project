import matplotlib.pyplot as plt
 
 
 
# figure 5
#  1-NNS
#x = [1000, 2000, 5000, 10000, 19000,61265]
#y = [5.72673, 10.8098, 20.0771, 33.1481, 46.2843, 89.5726]
 
# 10-NNS
x = [1000, 2000, 5000, 10000, 19000,61265]
y = [7.42543, 13.7027, 26.2823, 44.3193, 62.8308, 124.947]
 
plt.xlabel('Number of database points')
plt.ylabel('Disk accesses')
plt.plot(x, y, linestyle='--', marker='o')
plt.show()
 
 
# figure 7
# 1-NN
# x = [8, 16, 32]
# y = [13.4505, 35.6256, 88.1041]
 
# figure 7
# 10-NN
x = [8, 16, 32]
y = [14.8859, 43.8368, 120.718]
 
plt.xlabel('Dimensions')
plt.ylabel('Disk accesses')
plt.plot(x, y, linestyle='--', marker='o')
plt.show()
