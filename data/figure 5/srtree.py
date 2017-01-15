import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

dsk_acc_mult = 1
 
# figure 5
#  1-NNS
x = [1000, 2000, 5000, 10000, 19000]
y = [6.53106, 11.2064, 19.1603, 31.998, 52.2124]
y_e2p = [5,5,5,6,7] * dsk_acc_mult
y_e5p = [4,3,4,4,5] * dsk_acc_mult
y_e10p = [3,2,3,3,3] * dsk_acc_mult
y_e20p = [2,2,2,2,2] * dsk_acc_mult

plt.xlabel('Number of database points')
plt.ylabel('Disk accesses')
plt.plot(x, y, linestyle='--', marker='o',color='blue')
plt.plot(x, y_e2p, marker='s',color='red')
plt.plot(x, y_e5p, marker='s',color='yellow')
plt.plot(x, y_e10p, marker='s',color='purple')
plt.plot(x, y_e20p, marker='s',color='green')

bp = mpatches.Patch(color='blue', label='SR-tree')
rp = mpatches.Patch(color='red', label='LSH e=0.02')
yp = mpatches.Patch(color='yellow', label='LSH e=0.05')
pp = mpatches.Patch(color='purple', label='LSH e=0.1')
gp = mpatches.Patch(color='green', label='LSH e=0.2')

plt.legend(handles=[bp,rp,yp,pp,gp],loc=2)
plt.title("Approximate 1-NNS")
plt.show()


# 10-NNS
y = [8.28858, 14.2164, 24.7756, 41.7876, 68.6433]
y_e2p = [7,6,7,8,10] * dsk_acc_mult
y_e5p = [6,5,4,5,6] * dsk_acc_mult
y_e10p = [4,3,3,4,4] * dsk_acc_mult
y_e20p = [2,2,2,2,3] * dsk_acc_mult

plt.xlabel('Number of database points')
plt.ylabel('Disk accesses')
plt.plot(x, y, linestyle='--', marker='o',color='blue')
plt.plot(x, y_e2p, marker='s',color='red')
plt.plot(x, y_e5p, marker='s',color='yellow')
plt.plot(x, y_e10p, marker='s',color='purple')
plt.plot(x, y_e20p, marker='s',color='green')

bp = mpatches.Patch(color='blue', label='SR-tree')
rp = mpatches.Patch(color='red', label='LSH e=0.02')
yp = mpatches.Patch(color='yellow', label='LSH e=0.05')
pp = mpatches.Patch(color='purple', label='LSH e=0.1')
gp = mpatches.Patch(color='green', label='LSH e=0.2')

plt.legend(handles=[bp,rp,yp,pp,gp],loc=2)
plt.title("Approximate 10-NNS")
plt.show()
