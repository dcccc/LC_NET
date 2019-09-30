#coding:utf8
# import torch
from torch import nn,optim,tensor,LongTensor
from torch.nn.utils.rnn import pack_padded_sequence,pad_sequence,PackedSequence,pack_sequence
from torch.autograd import Variable
from torch.nn.functional import tanh
import torch.nn.init as init
from torch.optim.lr_scheduler import StepLR
import numpy as np

import torch

class cnn(nn.Module):
	def __init__(self,batch_size,in_dim1,in_dim2,in_dim3,n_layer):
		super(cnn,self).__init__()

		self.embed=torch.nn.Embedding(85,32)
		init.xavier_uniform_(self.embed.weight,gain=1.0)

		self.lstm1 = nn.LSTM(in_dim1, in_dim2, n_layer, batch_first=True)

		init.xavier_normal_(self.lstm1.all_weights[0][0], gain=np.sqrt(2.0))
		init.xavier_normal_(self.lstm1.all_weights[1][1], gain=np.sqrt(2.0))
		init.xavier_normal_(self.lstm1.all_weights[1][0], gain=np.sqrt(2.0))
		init.xavier_normal_(self.lstm1.all_weights[1][1], gain=np.sqrt(2.0))


		self.batchnorm2=nn.BatchNorm1d(batch_size)

		self.lstm2 = nn.LSTM(in_dim2, in_dim3, n_layer, batch_first=True)
		init.xavier_normal_(self.lstm2.all_weights[0][0], gain=np.sqrt(2.0))
		init.xavier_normal_(self.lstm2.all_weights[1][1], gain=np.sqrt(2.0))
		init.xavier_normal_(self.lstm2.all_weights[1][0], gain=np.sqrt(2.0))
		init.xavier_normal_(self.lstm2.all_weights[1][1], gain=np.sqrt(2.0))


		self.batchnorm3=nn.BatchNorm1d(batch_size)

		self.linear1=nn.Linear(in_dim3,128)
		init.xavier_uniform_(self.linear1.weight, gain=np.sqrt(2.0))
		init.constant_(self.linear1.bias, 0.1)
		self.dp1=nn.Dropout(p=0.5)
		self.relu1=nn.ReLU(inplace=True)


		self.linear2=nn.Linear(128,64)
		init.xavier_uniform_(self.linear2.weight, gain=np.sqrt(2.0))
		init.constant_(self.linear2.bias, 0.1)
		self.dp2=nn.Dropout(p=0.5)
		self.relu2=nn.ReLU(inplace=True)
		
		
		self.linear22=nn.Linear(64,64)
		init.xavier_uniform_(self.linear22.weight, gain=np.sqrt(2.0))
		init.constant_(self.linear22.bias, 0.1)
		self.dp22=nn.Dropout(p=0.5)
		self.relu22=nn.ReLU(inplace=True)
		

		self.linear3=nn.Linear(64,32)
		init.xavier_uniform_(self.linear3.weight, gain=np.sqrt(2.0))
		init.constant_(self.linear3.bias, 0.1)
		self.dp3=nn.Dropout(p=0.5)
		self.relu3=nn.ReLU(inplace=True)
		

		self.linear4=nn.Linear(32, 32)
		init.xavier_uniform_(self.linear4.weight, gain=np.sqrt(2.0))
		init.constant_(self.linear4.bias, 0.1)
		self.relu4=nn.ReLU(inplace=True)
		
		self.linear5=nn.Linear(32,16)
		init.xavier_uniform_(self.linear2.weight, gain=np.sqrt(2.0))
		init.constant_(self.linear2.bias, 0.1)

		self.relu5=nn.ReLU(inplace=True)
		



	def forward(self,neighbor_atom_batch,neighbors_distan_batch,structure_num_bacth,dp=1):

		
		h1   = torch.from_numpy(np.ones((4,len(neighbor_atom_batch)//20,64))).float()
		c1   = torch.from_numpy(np.ones((4,len(neighbor_atom_batch)//20,64))).float()

		data = torch.from_numpy(neighbor_atom_batch).long()
		data = self.embed(data)
		
		data = data*torch.from_numpy(neighbor_atom_batch).float().view(-1,1)
		data = data.view(-1,20,32)

		_,(data1,data2)=self.lstm1(data,(h1,c1))
		data = torch.mean(data1,dim=0,keepdim=False)+torch.mean(data2,dim=0,keepdim=False)


		length_cum=np.cumsum(np.append([0],structure_num_bacth))

		data = [data[length_cum[i]:length_cum[i+1]] for i in range(len(length_cum)-1) ]
		data = pad_sequence(data)

		h2   = torch.from_numpy(np.ones((4,data.size()[1],128))).float()
		c2   = torch.from_numpy(np.ones((4,data.size()[1],128))).float()

		data = self.batchnorm2(data).permute(1,0,2)

		_,(data1,data2)=self.lstm2(data,(h2,c2))
		data = torch.mean(data2,dim=0,keepdim=False) #+torch.mean(data2,dim=0,keepdim=False)

		data = self.batchnorm3(data.permute(1,0)).permute(1,0)
		


		data = self.linear1(data)
		data = self.relu1(data)
		# if dp==1:
		# 	data = self.dp1(data)
			
		data = self.linear2(data)
		data = self.relu2(data)
		# if dp==1:
		# 	data = self.dp2(data)
		
		data = self.batchnorm3(data.permute(1,0)).permute(1,0)

		data = self.linear22(data)
		data = self.relu22(data)
		# if dp==1:
		# 	data = self.dp22(data)
			
		
		data = self.linear3(data)
		data = self.relu3(data)
		# if dp==1:
		# 	data = self.dp3(data)
		
		data = self.linear4(data)
		data = self.relu4(data)
		
		data = self.linear5(data)

		data = torch.sum(data,1)
		data = self.relu5(data)

		return(data)



def data_batch(batch_num,batch_size,neighbor_atom,neighbors_distan,structure_num,label):

	if batch_num == 0:

		atom_num_line1 = 0
		atom_num_line2 = np.sum(structure_num[:batch_size])*20
		
		neighbor_atom_batch    = neighbor_atom[atom_num_line1:atom_num_line2]
		neighbors_distan_batch = neighbors_distan[atom_num_line1:atom_num_line2]
		structure_num_bacth    = structure_num[batch_num*batch_size:(batch_num+1)*batch_size]
		label_list_bacth       = label[batch_num*batch_size:(batch_num+1)*batch_size]


	elif batch_num < len(label)//batch_size:

		atom_num_line1 = np.sum(structure_num[:batch_num*batch_size])*20
		atom_num_line2 = np.sum(structure_num[:(batch_num+1)*batch_size])*20
	
		neighbor_atom_batch    = neighbor_atom[atom_num_line1:atom_num_line2]
		neighbors_distan_batch = neighbors_distan[atom_num_line1:atom_num_line2]
		structure_num_bacth    = structure_num[batch_num*batch_size:(batch_num+1)*batch_size]
		label_list_bacth       = label[batch_num*batch_size:(batch_num+1)*batch_size]
	
	else:
		return(False)

	# print(neighbor_atom_batch)
	# print(atom_num_line1 ,atom_num_line2)
	return([neighbor_atom_batch,neighbors_distan_batch,structure_num_bacth,label_list_bacth])

	
	

		
test_neighbor_atom     = np.genfromtxt("test_atom_neighbors_data", dtype=np.int32).reshape((-1,))
test_neighbors_distan  = np.genfromtxt("test_neighbors_distan_data", dtype=np.float64).reshape((-1,))
test_structure_num     = np.genfromtxt("test_structure_atom_num_data", dtype=np.int32).reshape((-1,))
test_e_gap             = open("test_label_of_qe", "r").readlines()
test_e_gap             = np.array([float(line.split()[0]) for line in test_e_gap])*3.0

train_neighbor_atom    = np.genfromtxt("train_atom_neighbors_data", dtype=np.int32).reshape((-1,))
train_neighbors_distan = np.genfromtxt("train_neighbors_distan_data", dtype=np.float64).reshape((-1,))
train_structure_num    = np.genfromtxt("train_structure_atom_num_data", dtype=np.int32).reshape((-1,))
train_e_gap            = open("train_label_of_qe", "r").readlines()
train_e_gap            = np.array([float(line.split()[0]) for line in train_e_gap])*3.0

batch_size=32

train_num=len(train_e_gap)//batch_size
test_num =len( test_e_gap)//batch_size

rnn_model = torch.load("20_cnn.pkl")


train_error=open("train_error","w")
test_error = open("test_error","w")
train_error.write("egap_dft   egap_pre   error\n")
test_error.write("egap_dft   egap_pre   error\n")

with torch.no_grad():
	
	for i in range(train_num):
		data_set    = data_batch(i,batch_size,train_neighbor_atom,train_neighbors_distan,train_structure_num,train_e_gap)
		
		out         = rnn_model(data_set[0],data_set[1],data_set[2])
		e_gap_batch = torch.from_numpy(data_set[3]).float()
		
		error       = out-e_gap_batch
		for j in range(len(error)):
			train_error.write("%6.5f     %6.5f     %6.5f\n"  %(e_gap_batch[j],out[j],error[j]))
		del data_set,out,e_gap_batch,error
	
	
	for i in range(test_num):
	
	
		data_set    = data_batch(i,batch_size,test_neighbor_atom,test_neighbors_distan,test_structure_num,test_e_gap)
		
		out         = rnn_model(data_set[0],data_set[1],data_set[2])
		e_gap_batch = torch.from_numpy(data_set[3]).float()
		
		error       = out-e_gap_batch
		for j in range(len(error)):
			test_error.write("%6.5f     %6.5f     %6.5f\n"  %(e_gap_batch[j],out[j],error[j]))
	
		del data_set,out,e_gap_batch,error

