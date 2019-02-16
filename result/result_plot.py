import numpy as np 
import os

from matplotlib import pyplot as plt


error_list=np.genfromtxt("error_result", dtype=np.float64).reshape((-1,))
e_gap=np.genfromtxt("label_of_qe", dtype=np.float64).reshape((-1,))

e_pre=error_list+e_gap[:28600]

plt.scatter(e_gap[:25725],e_pre[:25725],     s=3,marker="o",
            alpha=0.7, label = 'train')
plt.scatter(e_gap[25725:28600],e_pre[25725:],s=3,marker="^",
            alpha=0.7, label = 'text')

plt.xlabel('Egap(DFT)')			
plt.ylabel('Egap(predicted)')		
plt.legend('train test')
plt.show()


data=np.genfromtxt("ee.txt", dtype=np.float64)

plt.scatter(np.arange(80),data[:,1].T,     s=20,marker="o",
            alpha=0.7, label = 'train')
plt.scatter(np.arange(80),data[:,2].T,s=20,marker="^",
            alpha=0.7, label = 'text')

plt.xlabel('epoch')			
plt.ylabel('MAE')		
plt.legend('train test')
plt.show()		
			
			
