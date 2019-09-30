import numpy as np 
import os

from matplotlib import pyplot as plt


# train_error=np.genfromtxt("train_error", dtype=np.float64).T
# test_error =np.genfromtxt("test_error",  dtype=np.float64).T

# plt.scatter(test_error[0], test_error[1],      s=3,marker="^",
            # alpha=0.7, label = 'test')

# plt.scatter(train_error[0],train_error[1],     s=3,marker="o",
            # alpha=0.7, label = 'train')

# plt.xlabel('epoch')			
# plt.ylabel('MAE')		
# plt.legend('train test')
# plt.show()		


#误差分布直方图
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.hist(sorted(test_error[2,1:]),20,color='blue',alpha=0.6,rwidth=0.9,)
# ax.hist(sorted(train_error[2,1:]),20,color='blue',alpha=0.6,rwidth=0.9,)
# plt.show()

train_error=np.genfromtxt("epoch.txt", dtype=np.float64).T

plt.scatter(np.arange(66),train_error[4],     s=20,marker="^",            alpha=0.7, label = 'test')		
plt.scatter(np.arange(66),train_error[3],     s=20,marker="o",            alpha=0.7, label = 'train')


plt.xlabel('epoch')			
plt.ylabel('MAE')		
# plt.legend('train test')
plt.show()	