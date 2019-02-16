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




rnn_model = torch.load("47_cnn.pkl")
ceriterion = nn.MSELoss()
opt        = optim.Adam( rnn_model.parameters(), lr= 0.001)


error_list=open("error_result","a")


for i in range(1144):
	data_set    = data_batch(i,neighbor_atom,neighbors_distan,structure_num,e_gap)
	

	out         = rnn_model(data_set[0],data_set[1],data_set[2],dp=0)
	e_gap_batch = torch.from_numpy(data_set[3]).float()
	loss        = ceriterion(out.view(25,),e_gap_batch)
	run_loss    = loss.item()

	
	error       = out-e_gap_batch


	del loss
	del run_loss

	for line in error :
		error_list.write("%f\n" %(line/3) )
	error_list.flush()
	
	del error
	del out
	



