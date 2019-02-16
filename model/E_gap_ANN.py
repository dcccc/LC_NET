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
	def __init__(self,in_dim1,in_dim2,in_dim3,n_layer):
		super(cnn,self).__init__()

		self.embed=torch.nn.Embedding(85,32)
		init.xavier_uniform_(self.embed.weight,gain=1.0)

		self.lstm1 = nn.LSTM(in_dim1, in_dim2, n_layer, batch_first=True)

		init.xavier_normal_(self.lstm1.all_weights[0][0], gain=np.sqrt(2.0))
		init.xavier_normal_(self.lstm1.all_weights[1][1], gain=np.sqrt(2.0))
		init.xavier_normal_(self.lstm1.all_weights[1][0], gain=np.sqrt(2.0))
		init.xavier_normal_(self.lstm1.all_weights[1][1], gain=np.sqrt(2.0))


		self.batchnorm2=nn.BatchNorm1d(25)

		self.lstm2 = nn.LSTM(in_dim2, in_dim3, n_layer, batch_first=True)
		init.xavier_normal_(self.lstm2.all_weights[0][0], gain=np.sqrt(2.0))
		init.xavier_normal_(self.lstm2.all_weights[1][1], gain=np.sqrt(2.0))
		init.xavier_normal_(self.lstm2.all_weights[1][0], gain=np.sqrt(2.0))
		init.xavier_normal_(self.lstm2.all_weights[1][1], gain=np.sqrt(2.0))


		self.batchnorm3=nn.BatchNorm1d(25)

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
		



	def forward(self,neighbor_atom_batch,neighbors_distan_batch,structure_num_bacth):

		
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
		data = torch.mean(data2,dim=0,keepdim=False)

		data = self.batchnorm3(data.permute(1,0)).permute(1,0)
		


		data = self.linear1(data)
		data = self.relu1(data)

			
		data = self.linear2(data)
		data = self.relu2(data)

		
		data = self.batchnorm3(data.permute(1,0)).permute(1,0)

		data = self.linear22(data)
		data = self.relu22(data)

			
		
		data = self.linear3(data)
		data = self.relu3(data)

		
		data = self.linear4(data)
		data = self.relu4(data)
		
		data = self.linear5(data)

		data = torch.sum(data,1)
		data = self.relu5(data)

		return(data)



def data_batch(batch_num,neighbor_atom,neighbors_distan,structure_num,label):

	if batch_num == 0:

		atom_num_line1 = 0
		atom_num_line2 = np.sum(structure_num[:25])*20
		
		neighbor_atom_batch    = neighbor_atom[atom_num_line1:atom_num_line2]
		neighbors_distan_batch = neighbors_distan[atom_num_line1:atom_num_line2]
		structure_num_bacth    = structure_num[batch_num*25:(batch_num+1)*25]
		label_list_bacth       = label[batch_num*25:(batch_num+1)*25]


	elif batch_num < 1900:

		atom_num_line1 = np.sum(structure_num[:batch_num*25])*20
		atom_num_line2 = np.sum(structure_num[:(batch_num+1)*25])*20
	
		neighbor_atom_batch    = neighbor_atom[atom_num_line1:atom_num_line2]
		neighbors_distan_batch = neighbors_distan[atom_num_line1:atom_num_line2]
		structure_num_bacth    = structure_num[batch_num*25:(batch_num+1)*25]
		label_list_bacth       = label[batch_num*25:(batch_num+1)*25]
	
		
	else:

		atom_num_line1 = np.sum(structure_num[:batch_num*25])*20
		atom_num_line2 = 0
	
		neighbor_atom_batch    = neighbor_atom[atom_num_line1:]
		neighbors_distan_batch = neighbors_distan[atom_num_line1:]
		structure_num_bacth    = structure_num[batch_num*25:]
		label_list_bacth       = label[batch_num*25:]


	return([neighbor_atom_batch,neighbors_distan_batch,structure_num_bacth,label_list_bacth])

	
	

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

		
		
neighbor_atom    = np.genfromtxt("atom_neighbors_data", dtype=np.int32).reshape((-1,))
neighbors_distan = np.genfromtxt("neighbors_distan_data", dtype=np.float64).reshape((-1,))
structure_num    = np.genfromtxt("structure_atom_num_data", dtype=np.int32).reshape((-1,))
e_gap            = np.genfromtxt("label_of_qe", dtype=np.float64).reshape((-1,))*3



rnn_model  = cnn(32,64,128,4)
# rnn_model = torch.load("cnn.pkl")
ceriterion = nn.MSELoss()

opt        = optim.Adam( rnn_model.parameters(), lr= 0.001)



result = open("result","a")

data_set=[]

for epoch in range(80):

	e_gap_mse_total     = 0.0
	e_gap_mse_test      = 0.0

	e_gap_mae_total     = 0.0
	e_gap_mae_test      = 0.0


	for i in range(1144):


		data_set    = data_batch(i,neighbor_atom,neighbors_distan,structure_num,e_gap)
		
		adjust_learning_rate(opt,epoch)


		if i<1029:
		
			out         = rnn_model(data_set[0],data_set[1],data_set[2])
			e_gap_batch = torch.from_numpy(data_set[3]).float()
			loss        = ceriterion(out.view(25,),e_gap_batch)

			run_loss    = loss.item()
		

			
			error       = out-e_gap_batch

			e_gap_mse   = torch.sum( error*error)/25/9
			e_gap_mae   =  torch.sum(torch.abs(error))/25/3
	
			
			opt.zero_grad()
			loss.backward()
			opt.step()

			del loss
			e_gap_mse_total  +=e_gap_mse
			e_gap_mae_total  +=e_gap_mae




		if i>1029:
		
			out         = rnn_model(data_set[0],data_set[1],data_set[2])
			e_gap_batch = torch.from_numpy(data_set[3]).float()
			loss        = ceriterion(out.view(25,),e_gap_batch)

			run_loss    = loss.item()
		
			
			error       = out-e_gap_batch

			e_gap_mse   = torch.sum( error*error)/25/9
			e_gap_mae   =  torch.sum(torch.abs(error))/25/3


			del loss


			e_gap_mse_test  +=e_gap_mse
			e_gap_mae_test  +=e_gap_mae

	
	torch.save(rnn_model,str(epoch)+"_cnn.pkl")

	result.write("%d %4.3f  %4.3f  %4.3f  %4.3f\n"  %(epoch,e_gap_mse_total/1029,e_gap_mse_test/115,e_gap_mae_total/1029,e_gap_mae_test/115))
	result.flush()





