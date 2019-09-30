import numpy as np 


test_error=open("train_error","r").readlines()
test_label=open("train_label_of_qe","r").readlines()

test_list=[test_error[i+1].split()+test_label[i].split() for i in range(len(test_error)-1)]

test_list.sort(key=lambda x : x[-3])

n=0
nn=0
error=0

while(n<len(test_list)):
	if  n<len(test_list)-3 and test_list[n][-3]==test_list[n+3][-3] :
		test_pre=float(test_list[n][1])+float(test_list[n+1][1])+\
		         float(test_list[n+2][1])+float(test_list[n+3][1])
		test_dft=float(test_list[n][0])
		error+=abs(test_pre/3/4-test_dft/3)
		n+=4
	elif n<len(test_list)-2 and test_list[n][-3]==test_list[n+2][-3]  :
		test_pre=float(test_list[n][1])+float(test_list[n+1][1])+\
		         float(test_list[n+2][1])
		test_dft=float(test_list[n][0])
		error+=abs(test_pre/3/3-test_dft/3)
		n+=3
	elif n<len(test_list)-1 and test_list[n][-3]==test_list[n+1][-3]  :
		test_pre=float(test_list[n][1])+float(test_list[n+1][1])
		test_dft=float(test_list[n][0])
		error+=abs(test_pre/3/2-test_dft/3)
		n+=2
	else :
		test_pre=float(test_list[n][1])
		test_dft=float(test_list[n][0])
		error+=abs(test_pre/3-test_dft/3)
		n+=1
	nn+=1
		
print(nn,error/nn)
