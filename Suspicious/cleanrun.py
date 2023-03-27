# train a dominant detector

from pygod.models import DOMINANT
from pygod.models import ANOMALOUS
from pygod.models import MLPAE
from SuspiciousGCN import Both as SGCN
from SuspiciousGSage import Both as SSage
from SuspiciousGSage import Both as SSGC
from SuspiciousGAT import Both as SGAT
from torch_geometric.data import Data
import torch_geometric.utils as Utils
from scipy.sparse import data
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse
import scipy.io
from sklearn.metrics import roc_auc_score
from datetime import datetime
import argparse
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygod.generator import gen_contextual_outliers, gen_structural_outliers
from sklearn import model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import networkx
import torch
import torch.nn as nn
import torch.nn.functional as Fu
import torch.optim as optim
from torch.nn.parameter import Parameter

import time
import math
import random

# train a dominant detector
from pygod.models import DOMINANT
from torch_geometric.data import Data
from scipy.sparse import data
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse
import scipy.io
from sklearn.metrics import roc_auc_score
from datetime import datetime
import argparse
import gc

from pygod.utils.utility import validate_device
from pygod.models import AnomalyDAE
import csv
import os
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid 
from torch_geometric.datasets import Amazon 
from torch_geometric.data import Data 
import random
from pygod.models import DOMINANT
from pygod.metrics import eval_roc_auc
from pygod.models import AnomalyDAE
import csv
import os
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid 
from torch_geometric.datasets import Amazon 
import random
from pygod.models import DOMINANT

from pygod.metrics import eval_roc_auc
class LinearClassifier(nn.Module):
		def __init__(self, input_dim):
				super(LinearClassifier, self).__init__()
				
				self.fc = nn.Linear(input_dim, 1)
				
		def forward(self, x):
				x = self.fc(x)
				x = torch.sigmoid(x)
				return x

	
class GCN(nn.Module):
		def __init__(self, nfeat, nhid, nclass, dropout):
				super(GCN, self).__init__()
				self.nfeat = nfeat
				self.nhid = nhid
				self.nclass = nclass
				self.dropout = dropout
				
				# weight and bias between input and hidden layer
				self.weight_in_hid = Parameter(torch.FloatTensor(nfeat, nhid))
				self.bias_in_hid = Parameter(torch.FloatTensor(nhid))
				
				# weight and bias between hidden and output layer
				self.weight_hid_out = Parameter(torch.FloatTensor(nhid, nclass))
				self.bias_hid_out = Parameter(torch.FloatTensor(nclass))
				
				self.drop_layer = nn.Dropout(p=self.dropout)
				
				self.reset_parameters()
				
		def reset_parameters(self):
				torch.manual_seed(0)
				# use Glorot weight initialization
				stdv_in_hid = math.sqrt(6) / math.sqrt(self.nfeat + self.weight_in_hid.size(1))
				self.weight_in_hid.data.uniform_(-stdv_in_hid, stdv_in_hid)
				self.bias_in_hid.data.uniform_(-stdv_in_hid, stdv_in_hid)
				
				stdv_hid_out = math.sqrt(6) / math.sqrt(self.weight_in_hid.size(1) + self.weight_hid_out.size(1))
				self.weight_hid_out.data.uniform_(-stdv_hid_out, stdv_hid_out)
				self.bias_hid_out.data.uniform_(-stdv_hid_out, stdv_hid_out)
				
				
		def forward(self, x, adj):
				o_hidden = torch.mm(x, self.weight_in_hid)
				
				o_hidden = torch.spmm(adj, o_hidden)
				
				o_hidden = o_hidden + self.bias_in_hid
				
				o_hidden = Fu.relu(o_hidden)
				
				o_hidden = self.drop_layer(o_hidden)
				
				o_out = torch.mm(o_hidden, self.weight_hid_out)
				
				o_out = torch.spmm(adj, o_out)
				
				o_out = o_out + self.bias_hid_out
				
				return o_out
def get_top_k_indices(arr, k):
		indices = np.argpartition(arr, -k)[-k:]
		indices = indices[np.argsort(arr[indices])][::-1]
		return indices
def precision_at_k(y_true, y_score, k, pos_label=0):
		from sklearn.utils import column_or_1d
		from sklearn.utils.multiclass import type_of_target
		
		y_true_type = type_of_target(y_true)
		if not (y_true_type == "binary"):
				raise ValueError("y_true must be a binary column.")
		
		# Makes this compatible with various array types
		y_true_arr = column_or_1d(y_true)
		y_score_arr = column_or_1d(y_score)
		
		y_true_arr = y_true_arr == pos_label
		
		desc_sort_order = np.argsort(y_score_arr)[::-1]
		y_true_sorted = y_true_arr[desc_sort_order]
		y_score_sorted = y_score_arr[desc_sort_order]
		
		true_positives = y_true_sorted[:k].sum()
		
		return true_positives / k

def recall_at_k(y_true, y_score, k, pos_label=0):
		from sklearn.utils import column_or_1d
		from sklearn.utils.multiclass import type_of_target
		
		y_true_type = type_of_target(y_true)
		if not (y_true_type == "binary"):
				raise ValueError("y_true must be a binary column.")
		
		# Makes this compatible with various array types
		y_true_arr = column_or_1d(y_true)
		y_score_arr = column_or_1d(y_score)
		
		y_true_arr = y_true_arr == pos_label
		
		desc_sort_order = np.argsort(y_score_arr)[::-1]
		y_true_sorted = y_true_arr[desc_sort_order]
		y_score_sorted = y_score_arr[desc_sort_order]
		
		true_positives = y_true_sorted[:k].sum()
		
		return true_positives / y_true.sum()
class EarlyStopping:
		"""Early stops the training if validation loss doesn't improve after a given patience."""
		def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
				"""
				Args:
						patience (int): How long to wait after last time validation loss improved.
														Default: 7
						verbose (bool): If True, prints a message for each validation loss improvement. 
														Default: False
						delta (float): Minimum change in the monitored quantity to qualify as an improvement.
														Default: 0
						path (str): Path for the checkpoint to be saved to.
														Default: 'checkpoint.pt'
						trace_func (function): trace print function.
														Default: print						
				"""
				self.patience = patience
				self.verbose = verbose
				self.counter = 0
				self.best_score = None
				self.early_stop = False
				self.val_loss_min = np.Inf
				self.delta = delta
				self.path = path
				self.trace_func = trace_func
		def __call__(self, val_loss, model):

				score = -val_loss

				if self.best_score is None:
						self.best_score = score
						self.save_checkpoint(val_loss, model)
				elif score < self.best_score + self.delta:
						self.counter += 1
						# self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
						if self.counter >= self.patience:
								self.early_stop = True
				else:
						self.best_score = score
						self.save_checkpoint(val_loss, model)
						self.counter = 0

		def save_checkpoint(self, val_loss, model):
				'''Saves model when validation loss decrease.'''
				# if self.verbose:
						# self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).	Saving model ...')
				torch.save(model.state_dict(), self.path)
				self.val_loss_min = val_loss
def anomaly_score(node_embedding, c):
		# anomaly score of an instance is calculated by 
		# square Euclidean distance between the node embedding and the center c
		return torch.sum((node_embedding - c) ** 2)
def nor_loss(node_embedding_list, c):
		# normal loss is calculated by mean squared Euclidian distance of 
		# the normal node embeddings to hypersphere center c 
		s = 0
		num_node = node_embedding_list.size()[0]
		for i in range(num_node):
				s = s + anomaly_score(node_embedding_list[i], c)
		return s/num_node
def AUC_loss(anomaly_node_emb, normal_node_emb,c):
		# AUC_loss encourages the score of anomaly instances to be higher than those of normal instances
	s = 0
	num_anomaly_node = anomaly_node_emb.size()[0]
	num_normal_node = normal_node_emb.size()[0]
	anom=torch.empty(num_anomaly_node)
	norm=torch.empty(num_normal_node)
	for i in range(num_anomaly_node):
		anom[i]=anomaly_score(anomaly_node_emb[i],c)
	for j in range(num_normal_node):
		norm[j]=anomaly_score(normal_node_emb[j],c)
	for i in range(num_anomaly_node):
		for j in range(num_normal_node):
			
			# t_total = time.time()
			s1 = anom[i]
			s2 = norm[j]
			s = s + torch.sigmoid(s1 - s2)
			# print("Total time elapsed111: {:.4f}s".format(time.time() - t_total))
	# print("Total time elapsed112: {:.4f}s".format(time.time() - t_total2))
	if((num_anomaly_node * num_normal_node)==0):
		return s/((num_anomaly_node * num_normal_node)+1)	
	return s/(num_anomaly_node * num_normal_node) # check devide by zero
def objecttive_loss(anomaly_node_emb, normal_node_emb, c, regularizer=1):
	t_total = time.time()
	Nloss = nor_loss(normal_node_emb, c)
	# print("1:"+str(time.time() - t_total))
	t_total = time.time()
	AUCloss = AUC_loss(anomaly_node_emb, normal_node_emb,c)
	# print("2:"+str(time.time() - t_total))
	t_total = time.time()
	# AUCloss = AUC_loss(anomaly_node_emb, normal_node_emb, c)
	# AUCloss = eval_roc_auc(y, h)
	loss = Nloss - regularizer * AUCloss
	return loss
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='Cora', help='dataset name: Flickr/ACM/BlogCatalog')
	parser.add_argument('--hidden_dim', type=int, default=32, help='dimension of hidden embedding (default: 64)')
	parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
	parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
	parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
	parser.add_argument('--alpha', type=float, default=0.5, help='balance parameter')
	parser.add_argument('--device', default=0, type=int, help='cuda/cpu')
	parser.add_argument('--hidden_dim2', type=int, default=32, help='dimension of hidden embedding (default: 64)')
	parser.add_argument('--epoch2', type=int, default=100, help='Training epoch')
	parser.add_argument('--lr2', type=float, default=5e-3, help='learning rate')
	parser.add_argument('--dropout2', type=float, default=0.5, help='Dropout rate')
	parser.add_argument('--s_dens', type=float, default=0.1, help='Suspected sample density')
	parser.add_argument('--error', type=float, default=0, help='Suspected sample density')
	parser.add_argument('--anomaly', default='Out', help='anomaly type: Min/Out/MinOut')
	parser.add_argument('--training',default='suspicious')
	parser.add_argument('--anomalyrate', type=float, default=0.1)
	parser.add_argument('--vsanomalyrate', type=float,default=1.0)
	parser.add_argument('--model', default='SGCN')
	parser.add_argument('--klist', default=[100])
	# parser.add_argument('--trust', type=float, default=1., help='Trust')
	args = parser.parse_args()
	# os.environ["CUDA_VISIBLE_DEVICES"]=""
	
	F = open("ResultAll.csv","a")
	writer=csv.writer(F,delimiter=';')
	F2 = open("ResultPartial.csv","a")
	writer2=csv.writer(F2,delimiter=';')
	
	
	print(parser)
	
	
	
	if args.dataset=="Cora" :
		data = Planetoid('./data/'+args.dataset, args.dataset, transform=T.NormalizeFeatures())[0]
		data.y[data.y!=6] = 0
		data.y[data.y==6]=1
	elif args.dataset=="Citeseer" :
		data = Planetoid('./data/'+args.dataset, args.dataset, transform=T.NormalizeFeatures())[0]
		data.y[data.y!=0] = 10
		data.y[data.y==0]=1
		data.y[data.y==10]=0
	elif	args.dataset=="PubMed" :
		data = Planetoid('./data/'+args.dataset, args.dataset, transform=T.NormalizeFeatures())[0]
		data.y[data.y!=0] = 10
		data.y[data.y==0]=1
		data.y[data.y==10]=0
	elif	args.dataset=="Computers" :
		data = Amazon('./data/'+args.dataset, args.dataset, transform=T.NormalizeFeatures())[0]
		data.y[data.y!=9] = 0
		data.y[data.y==9]=1
	elif	args.dataset=="Photo" :
		data = Amazon('./data/'+args.dataset, args.dataset, transform=T.NormalizeFeatures())[0]
		data.y[data.y!=7] = 0
		data.y[data.y==7]=1
	elif args.dataset=="Disney" :
		# temp = networkx.read_graphml('./data/'+args.dataset+'/'+args.dataset+'.graphml')
		# data =Utils.from_networkx(temp,all)
		# my_data = np.genfromtxt('./data/'+args.dataset+'/'+args.dataset+'.true', delimiter=';')
		# my_data = pd.DataFrame(my_data).sort_values(by=0)
		data=torch.load(r'./data/disney.pt')
	elif args.dataset=="Books" :
		data=torch.load(r'./data/books.pt')
	elif args.dataset=="Enron" :
		data=torch.load(r'./data/enron.pt')
	elif args.dataset=="Synth20" :
		temp = networkx.read_graphml('./data/'+args.dataset+'/Graph_nodes_1000_edges_1322_dims_20.graphml')
			# print((temp.edges))
		order=indexes=np.arange(int(len(temp.nodes)))
		x=temp.nodes(data=True)
		df = pd.DataFrame(temp.nodes(data=True))
		res=df[1][:].items()
		res=list(res)
		data.x=np.array(res)
		data.xf=np.empty((0,20),float)
		for i in data.x :
			res=i[1].values()
			res=list(res)
			res=np.array(res)
			data.xf=np.append(data.xf,np.array([res]), axis=0)
		data.x=torch.FloatTensor(data.xf.astype(float))

		df = pd.DataFrame(temp.edges())
		data.edge_index=torch.IntTensor(np.array(df).astype(float).T).to(torch.int64)
		my_data = np.genfromtxt('./data/'+args.dataset+'/Graph_nodes_1000_edges_1322_dims_20.true', delimiter=';')
		my_data = pd.DataFrame(my_data).sort_values(by=0)
		data.y=torch.BoolTensor(np.array(my_data).astype(int)[:,1])
		
		data=Data(x=data.x,edge_index=data.edge_index,y=data.y)
	elif args.dataset=="Synth40" :
		temp = networkx.read_graphml('./data/'+args.dataset+'/Graph_nodes_1000_edges_1431_dims_40.graphml')
			# print((temp.edges))
		order=indexes=np.arange(int(len(temp.nodes)))
		x=temp.nodes(data=True)
		df = pd.DataFrame(temp.nodes(data=True))
		res=df[1][:].items()
		res=list(res)
		data.x=np.array(res)
		data.xf=np.empty((0,40),float)
		for i in data.x :
			res=i[1].values()
			res=list(res)
			res=np.array(res)
			data.xf=np.append(data.xf,np.array([res]), axis=0)
		data.x=torch.FloatTensor(data.xf.astype(float))

		df = pd.DataFrame(temp.edges())
		data.edge_index=torch.IntTensor(np.array(df).astype(float).T).to(torch.int64)
		my_data = np.genfromtxt('./data/'+args.dataset+'/Graph_nodes_1000_edges_1431_dims_40.true', delimiter=';')
		my_data = pd.DataFrame(my_data).sort_values(by=0)
		data.y=torch.BoolTensor(np.array(my_data).astype(int)[:,1])
		
		data=Data(x=data.x,edge_index=data.edge_index,y=data.y)
	elif args.dataset=="Synth60" :
		temp = networkx.read_graphml('./data/'+args.dataset+'/Graph_nodes_1000_edges_1361_dims_60.graphml')
			# print((temp.edges))
		order=indexes=np.arange(int(len(temp.nodes)))
		x=temp.nodes(data=True)
		df = pd.DataFrame(temp.nodes(data=True))
		res=df[1][:].items()
		res=list(res)
		data.x=np.array(res)
		data.xf=np.empty((0,60),float)
		for i in data.x :
			res=i[1].values()
			res=list(res)
			res=np.array(res)
			data.xf=np.append(data.xf,np.array([res]), axis=0)
		data.x=torch.FloatTensor(data.xf.astype(float))

		df = pd.DataFrame(temp.edges())
		data.edge_index=torch.IntTensor(np.array(df).astype(float).T).to(torch.int64)
		my_data = np.genfromtxt('./data/'+args.dataset+'/Graph_nodes_1000_edges_1361_dims_60.true', delimiter=';')
		my_data = pd.DataFrame(my_data).sort_values(by=0)
		data.y=torch.BoolTensor(np.array(my_data).astype(int)[:,1])
		
		data=Data(x=data.x,edge_index=data.edge_index,y=data.y)
	elif args.dataset=="Synth80" :
		temp = networkx.read_graphml('./data/'+args.dataset+'/Graph_nodes_1000_edges_1485_dims_80.graphml')
			# print((temp.edges))
		order=indexes=np.arange(int(len(temp.nodes)))
		x=temp.nodes(data=True)
		df = pd.DataFrame(temp.nodes(data=True))
		res=df[1][:].items()
		res=list(res)
		data.x=np.array(res)
		data.xf=np.empty((0,80),float)
		for i in data.x :
			res=i[1].values()
			res=list(res)
			res=np.array(res)
			data.xf=np.append(data.xf,np.array([res]), axis=0)
		data.x=torch.FloatTensor(data.xf.astype(float))

		df = pd.DataFrame(temp.edges())
		data.edge_index=torch.IntTensor(np.array(df).astype(float).T).to(torch.int64)
		my_data = np.genfromtxt('./data/'+args.dataset+'/Graph_nodes_1000_edges_1485_dims_80.true', delimiter=';')
		my_data = pd.DataFrame(my_data).sort_values(by=0)
		data.y=torch.BoolTensor(np.array(my_data).astype(int)[:,1])
		
		data=Data(x=data.x,edge_index=data.edge_index,y=data.y)
		
	elif args.dataset=="Weibo" :
		data=torch.load(r'./data/weibo.pt')
	elif args.dataset=="reddit" :
		data=torch.load(r'./data/reddit.pt')
	elif args.dataset=="FlickrS" :
		data=torch.load(r'./data/inj_flickr.pt')
		# data.y=data.y >0
		data.y=data.y ==1
	elif args.dataset=="CoraS" :
		data=torch.load(r'./data/inj_cora.pt')
		data.y=data.y >1
	elif args.dataset=="AmazonS" :
		data=torch.load(r'./data/inj_amazon.pt')
		data.y=data.y ==1
	elif args.dataset=="FlickrC" :
		data=torch.load(r'./data/inj_flickr.pt')
		# data.y=data.y >0
		data.y=data.y ==2
	elif args.dataset=="CoraC" :
		data=torch.load(r'./data/inj_cora.pt')
		data.y=data.y ==2
	elif args.dataset=="AmazonC" :
		data=torch.load(r'./data/inj_amazon.pt')
		data.y=data.y ==2
	elif args.dataset=="Gen100" :
		data=torch.load(r'./data/gen_100.pt')
	elif args.dataset=="Gen500" :
		data=torch.load(r'./data/gen_500.pt')
	elif args.dataset=="Gen1000" :
		data=torch.load(r'./data/gen_1000.pt')
	elif args.dataset=="Gen5000" :
		data=torch.load(r'./data/gen_5000.pt')
	elif args.dataset=="Gen10000" :
		data=torch.load(r'./data/gen_10000.pt')
	# data, ya = gen_contextual_outliers(data, n=int(len(data.y)*0.10), k=50)
	# data, ys = gen_structural_outliers(data, m=int(len(data.y)*0.001), n=15)
	if args.anomaly!="MinOnly":
		 data, ya = gen_contextual_outliers(data, n=int(0.1*data.x.shape[0]/2), k=250)
		 data, ys = gen_structural_outliers(data, m=int(math.sqrt(0.1*data.x.shape[0]/2)), n=int(math.sqrt(0.1*data.x.shape[0]/2)))
	if args.anomaly=="Out":
		data.y = torch.logical_or(ys, ya).int()
	elif args.anomaly=="Con":
		data.y=ya
	elif args.anomaly=="Str":
		data.y=ys	 
	elif args.anomaly=="Min":
		data.y
	elif args.anomaly=="All":
		data.y = torch.logical_or(data.y, ya).int()
		data.y = torch.logical_or(data.y, ys).int()
	
	# print(data.id)
	Norm=np.empty(0)
	Susp=np.empty(0)
	NormSubSusp=np.empty(0)
	NormDivSusp=np.empty(0)
	Rec={}
	Acc={}
	for k in args.klist:
		Rec[k]=np.empty(0)
		Acc[k]=np.empty(0)
		
	print(Rec)
	T=np.empty(0)
	write2=np.empty(0)	
	write2=np.append(write2,args.dataset)
	write2=np.append(write2,args.s_dens)
	write2=np.append(write2,args.epoch)
	write2=np.append(write2,args.epoch2)
	write2=np.append(write2,args.dropout)
	write2=np.append(write2,args.dropout2)
	write2=np.append(write2,args.hidden_dim)
	write2=np.append(write2,args.hidden_dim2)
	write2=np.append(write2,args.alpha)               
	write2=np.append(write2,args.error)
	write2=np.append(write2,args.anomaly)
	write2=np.append(write2,args.vsanomalyrate)
	write2=np.append(write2,args.anomalyrate)
	write2=np.append(write2,args.model)
	write2=np.append(write2,args.klist)
	path="./"+str(args.dataset)+"/"+str(args.s_dens)+"/"+str(args.epoch)+"/"+str(args.epoch2)+"/"+str(args.lr)+'/'+str(args.lr2)
	isExist = os.path.exists(path)
	if not isExist:
		os.makedirs(path)	           
	device = validate_device(args.device)
	for j in range(0,10):		
		print(str(j))
		indexes=np.arange(int(len(data.y)))	
		normal=np.empty(0)
		suspicious=np.empty(0)
		index_labeled = np.empty(0)
		
		for i in range(1,int(len(data.y)*args.s_dens)):
			index_labeled=np.append(index_labeled,i)				
		#---------------------------------------------------------------------new block------------------------------------------------
		x = pd.DataFrame(data.x.numpy())
		y = pd.DataFrame(data.y.numpy())
		node_data=x
		feature_names = ["w_{}".format(ii) for ii in range(x.shape[1])]
		column_names =	feature_names + ["subject"]
		node_data.columns=feature_names
		node_data["label"]=y
		if args.training=="norm":
			unlabeled_data, labeled_data, index_unlabeled, index_labeled = train_test_split(node_data, indexes, test_size=0.1, stratify=data.y, random_state=j)
			index_labeled=index_labeled.astype(int)
			edge_list= pd.DataFrame(data.edge_index.numpy().T)
			edge_list.columns=['target','source']
			num_node, num_feature = node_data.shape[0], node_data.shape[1]-1
			node_index = np.array(node_data.index)
			index_unlabeled = np.delete(indexes,index_labeled.astype(int))
			unlabeled_data = np.take(node_data,index_unlabeled,axis=0)
			labeled_data = np.take(node_data,index_labeled,axis=0)
			train_data=labeled_data
			index_normal_train = np.numpy(0)
			index_anomaly_train = np.numpy(0)
			error=100-args.error
			for i in index_labeled:
				r=random.randrange(100)
				if node_data.iloc[i.astype(int)]['label'] == 0:
					if(r<error):
						index_normal_train.append(i)
					else:
						index_anomaly_train.append(i)
				elif node_data.iloc[i]['label'] == 1:
					if(r<error):
						index_anomaly_train.append(i)
					else:
						index_normal_train.append(i)	
			index_normal_train=np.append(index_unlabeled,index_normal_train)

		elif args.training=="suspicious":
			unlabeled_data, labeled_data, index_unlabeled, index_labeled = train_test_split(node_data, indexes, test_size=args.anomalyrate, stratify=node_data['label'], random_state=j)
			
			index_labeled=index_labeled.astype(int)
			edge_list= pd.DataFrame(data.edge_index.numpy().T)
			edge_list.columns=['target','source']
			num_node, num_feature = node_data.shape[0], node_data.shape[1]-1
			node_index = np.array(node_data.index)
			index_unlabeled = np.delete(indexes,index_labeled.astype(int))
			unlabeled_data = np.take(node_data,index_unlabeled,axis=0)
			
			index_normal_train = []
			index_anomaly_train = []
			indexes=np.arange(int(len(data.y)))
			for i in index_labeled:
				if node_data.iloc[i.astype(int)]['label'] == 0:
					index_normal_train.append(i)
				else :
					index_anomaly_train.append(i)
			j=0
			k=0
			l=len(index_anomaly_train)
			# print(index_normal_train)
			while l/(l+k)>args.vsanomalyrate:
				if(j>=len(index_normal_train)):
					j=0
				index_anomaly_train.append(index_normal_train[j])
				index_normal_train=np.delete(index_normal_train,j)
				j=j+1
				k=k+1	  
			
		vn = []
		vs = []
		sample_s =torch.LongTensor(index_anomaly_train)
		sample_n =torch.LongTensor(index_normal_train)
		if args.model=="SGCN":
			t_total = time.time()	
			model = SGCN(Sample_n=sample_n, Sample_s= sample_s,epoch=args.epoch,hid_dim=args.hidden_dim,gpu=args.device,dropout=args.dropout,alpha=args.alpha,batch_size=0)
			model.fit(data)
			Rn = model.decision_function(data)
			model = SGCN(Sample_n=sample_s, Sample_s= sample_n,epoch=args.epoch,hid_dim=args.hidden_dim,gpu=args.device,dropout=args.dropout,alpha=args.alpha)
			model.fit(data)
			Rs = model.decision_function(data)
			Rn=(Rn-Rn.min())/(Rn.max()-Rn.min())
			Rs=(Rs-Rs.min())/(Rs.max()-Rs.min())+0.0001
			Rn=np.nan_to_num(Rn)
			Rs=np.nan_to_num(Rs)
			T=np.append(T,time.time() - t_total)
			pd.DataFrame(Rn).to_csv("./result/Rn"+args.dataset+"_"+args.anomaly+"_"+str(args.vsanomalyrate)+"_"+str(args.anomalyrate)+"_"+args.model+"_"+str(j)+".csv",index=False,header=False)
			pd.DataFrame(Rs).to_csv("./result/Rs"+args.dataset+"_"+args.anomaly+"_"+str(args.vsanomalyrate)+"_"+str(args.anomalyrate)+"_"+args.model+"_"+str(j)+".csv",index=False,header=False)
			pd.DataFrame(data.y.numpy()).to_csv("./result/y"+args.dataset+"_"+args.anomaly+"_"+str(args.vsanomalyrate)+"_"+str(args.anomalyrate)+"_"+args.model+"_"+str(j)+".csv",index=False,header=False)
			auc_score = eval_roc_auc(data.y.numpy()[index_unlabeled], Rn[index_unlabeled])
			Norm=np.append(Norm,auc_score)
			print('AUC Score Norm:', auc_score)
			auc_score = eval_roc_auc(data.y.numpy()[index_unlabeled], -Rs[index_unlabeled])
			Susp=np.append(Susp,auc_score)
			print('AUC Score Susp:', auc_score)
			auc_score = eval_roc_auc(data.y.numpy()[index_unlabeled], Rn[index_unlabeled]-Rs[index_unlabeled])
			NormSubSusp=np.append(NormSubSusp,auc_score)
			print('AUC Score Norm-Susp:', auc_score)
			auc_score = eval_roc_auc(data.y.numpy()[index_unlabeled], np.nan_to_num(Rn[index_unlabeled]/Rs[index_unlabeled]))
			NormDivSusp=np.append(NormDivSusp,auc_score)
			print('AUC Score Norm/Susp:', auc_score)
			for k in args.klist:
				print(k)
				indices = get_top_k_indices( np.nan_to_num(Rn[index_unlabeled]/Rs[index_unlabeled]), k)
				print(data.y.numpy()[indices].sum())
				print(data.y.numpy().sum())
				recall= data.y.numpy()[indices].sum()/data.y.numpy().sum()
				accuracy= data.y.numpy()[indices].sum()/k
				print('Accuracy@'+str(k)+':', accuracy)
				print('Recall@'+str(k)+':', recall)
				Acc[k]=np.append(Acc[k],accuracy)
				Rec[k]=np.append(Rec[k],recall)				
		if args.model=="SSGC":
			t_total = time.time()	
			model = SSGC(Sample_n=sample_n, Sample_s= sample_s,epoch=args.epoch,hid_dim=args.hidden_dim,gpu=args.device,dropout=args.dropout,alpha=args.alpha,batch_size=0)
			model.fit(data)
			Rn = model.decision_function(data)
			model = SSGC(Sample_n=sample_s, Sample_s= sample_n,epoch=args.epoch,hid_dim=args.hidden_dim,gpu=args.device,dropout=args.dropout,alpha=args.alpha)
			model.fit(data)
			Rs = model.decision_function(data)
			Rn=(Rn-Rn.min())/(Rn.max()-Rn.min())
			Rs=(Rs-Rs.min())/(Rs.max()-Rs.min())+0.0001
			Rn=np.nan_to_num(Rn)
			Rs=np.nan_to_num(Rs)
			T=np.append(T,time.time() - t_total)
			pd.DataFrame(Rn).to_csv("./result/Rn"+args.dataset+"_"+args.anomaly+"_"+str(args.vsanomalyrate)+"_"+str(args.anomalyrate)+"_"+args.model+"_"+str(j)+".csv",index=False,header=False)
			pd.DataFrame(Rs).to_csv("./result/Rs"+args.dataset+"_"+args.anomaly+"_"+str(args.vsanomalyrate)+"_"+str(args.anomalyrate)+"_"+args.model+"_"+str(j)+".csv",index=False,header=False)
			pd.DataFrame(data.y.numpy()).to_csv("./result/y"+args.dataset+"_"+args.anomaly+"_"+str(args.vsanomalyrate)+"_"+str(args.anomalyrate)+"_"+args.model+"_"+str(j)+".csv",index=False,header=False)
			auc_score = eval_roc_auc(data.y.numpy()[index_unlabeled], Rn[index_unlabeled])
			Norm=np.append(Norm,auc_score)
			print('AUC Score Norm:', auc_score)
			auc_score = eval_roc_auc(data.y.numpy()[index_unlabeled], -Rs[index_unlabeled])
			Susp=np.append(Susp,auc_score)
			print('AUC Score Susp:', auc_score)
			auc_score = eval_roc_auc(data.y.numpy()[index_unlabeled], Rn[index_unlabeled]-Rs[index_unlabeled])
			NormSubSusp=np.append(NormSubSusp,auc_score)
			print('AUC Score Norm-Susp:', auc_score)
			auc_score = eval_roc_auc(data.y.numpy()[index_unlabeled], np.nan_to_num(Rn[index_unlabeled]/Rs[index_unlabeled]))
			NormDivSusp=np.append(NormDivSusp,auc_score)
			print('AUC Score Norm/Susp:', auc_score)
			for k in args.klist:
				print(k)
				indices = get_top_k_indices( np.nan_to_num(Rn[index_unlabeled]/Rs[index_unlabeled]), k)
				print(data.y.numpy()[indices].sum())
				print(data.y.numpy().sum())
				recall= data.y.numpy()[indices].sum()/data.y.numpy().sum()
				accuracy= data.y.numpy()[indices].sum()/k
				print('Accuracy@'+str(k)+':', accuracy)
				print('Recall@'+str(k)+':', recall)
				Acc[k]=np.append(Acc[k],accuracy)
				Rec[k]=np.append(Rec[k],recall)				
		elif args.model=="SGAT":		 
			t_total = time.time()	
			model = SGAT(Sample_n=sample_n, Sample_s= sample_s,epoch=args.epoch,hid_dim=args.hidden_dim,gpu=args.device,dropout=args.dropout,alpha=args.alpha,batch_size=0)
			model.fit(data)
			Rn = model.decision_function(data)
			model = SGAT(Sample_n=sample_s, Sample_s= sample_n,epoch=args.epoch,hid_dim=args.hidden_dim,gpu=args.device,dropout=args.dropout,alpha=args.alpha)
			model.fit(data)
			Rs = model.decision_function(data)
			print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
			Rn=(Rn-Rn.min())/(Rn.max()-Rn.min())
			Rs=(Rs-Rs.min())/(Rs.max()-Rs.min())+0.001
			Rn=np.nan_to_num(Rn)
			Rs=np.nan_to_num(Rs)
			T=np.append(T,time.time() - t_total)
			pd.DataFrame(Rn).to_csv("./result/Rn"+args.dataset+"_"+args.anomaly+"_"+str(args.vsanomalyrate)+"_"+str(args.anomalyrate)+"_"+args.model+"_"+str(j)+".csv",index=False,header=False)
			pd.DataFrame(Rs).to_csv("./result/Rs"+args.dataset+"_"+args.anomaly+"_"+str(args.vsanomalyrate)+"_"+str(args.anomalyrate)+"_"+args.model+"_"+str(j)+".csv",index=False,header=False)
			pd.DataFrame(data.y.numpy()).to_csv("./result/y"+args.dataset+"_"+args.anomaly+"_"+str(args.vsanomalyrate)+"_"+str(args.anomalyrate)+"_"+args.model+"_"+str(j)+".csv",index=False,header=False)

			auc_score = eval_roc_auc(data.y.numpy()[index_unlabeled], Rn[index_unlabeled])
			Norm=np.append(Norm,auc_score)
			print('AUC Score Norm:', auc_score)
			auc_score = eval_roc_auc(data.y.numpy()[index_unlabeled], -Rs[index_unlabeled])
			Susp=np.append(Susp,auc_score)
			print('AUC Score Susp:', auc_score)
			auc_score = eval_roc_auc(data.y.numpy()[index_unlabeled], Rn[index_unlabeled]-Rs[index_unlabeled])
			NormSubSusp=np.append(NormSubSusp,auc_score)
			print('AUC Score Norm-Susp:', auc_score)
			auc_score = eval_roc_auc(data.y.numpy()[index_unlabeled], np.nan_to_num(Rn[index_unlabeled]/Rs[index_unlabeled]))
			NormDivSusp=np.append(NormDivSusp,auc_score)
			print('AUC Score Norm/Susp:', auc_score)
			for k in args.klist:
				print(k)
				indices = get_top_k_indices( np.nan_to_num(Rn[index_unlabeled]/Rs[index_unlabeled]), k)
				print(data.y.numpy().sum())
				recall= data.y.numpy()[indices].sum()/data.y.numpy().sum()
				accuracy= data.y.numpy()[indices].sum()/k
				print('Accuracy@'+str(k)+':', accuracy)
				print('Recall@'+str(k)+':', recall)
				Acc[k]=np.append(Acc[k],accuracy)
				Rec[k]=np.append(Rec[k],recall)
		elif args.model=="SSage":		 
			t_total = time.time()	
			model = SSage(Sample_n=sample_n, Sample_s= sample_s,epoch=args.epoch,hid_dim=args.hidden_dim,gpu=args.device,dropout=args.dropout,alpha=args.alpha,batch_size=0)
			model.fit(data)
			Rn = model.decision_function(data)
			model = SSage(Sample_n=sample_s, Sample_s= sample_n,epoch=args.epoch,hid_dim=args.hidden_dim,gpu=args.device,dropout=args.dropout,alpha=args.alpha)
			model.fit(data)
			Rs = model.decision_function(data)
			print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
			Rn=(Rn-Rn.min())/(Rn.max()-Rn.min())
			Rs=(Rs-Rs.min())/(Rs.max()-Rs.min())+0.001
			Rn=np.nan_to_num(Rn)
			Rs=np.nan_to_num(Rs)
			T=np.append(T,time.time() - t_total)
			pd.DataFrame(Rn).to_csv("./result/Rn"+args.dataset+"_"+args.anomaly+"_"+str(args.vsanomalyrate)+"_"+str(args.anomalyrate)+"_"+args.model+"_"+str(j)+".csv",index=False,header=False)
			pd.DataFrame(Rs).to_csv("./result/Rs"+args.dataset+"_"+args.anomaly+"_"+str(args.vsanomalyrate)+"_"+str(args.anomalyrate)+"_"+args.model+"_"+str(j)+".csv",index=False,header=False)
			pd.DataFrame(data.y.numpy()).to_csv("./result/y"+args.dataset+"_"+args.anomaly+"_"+str(args.vsanomalyrate)+"_"+str(args.anomalyrate)+"_"+args.model+"_"+str(j)+".csv",index=False,header=False)
			auc_score = eval_roc_auc(data.y.numpy()[index_unlabeled], Rn[index_unlabeled])
			Norm=np.append(Norm,auc_score)
			print('AUC Score Norm:', auc_score)
			auc_score = eval_roc_auc(data.y.numpy()[index_unlabeled], -Rs[index_unlabeled])
			Susp=np.append(Susp,auc_score)
			print('AUC Score Susp:', auc_score)
			auc_score = eval_roc_auc(data.y.numpy()[index_unlabeled], Rn[index_unlabeled]-Rs[index_unlabeled])
			NormSubSusp=np.append(NormSubSusp,auc_score)
			print('AUC Score Norm-Susp:', auc_score)
			auc_score = eval_roc_auc(data.y.numpy()[index_unlabeled], np.nan_to_num(Rn[index_unlabeled]/Rs[index_unlabeled]))
			NormDivSusp=np.append(NormDivSusp,auc_score)
			print('AUC Score Norm/Susp:', auc_score)
			for k in args.klist:
				print(k)
				indices = get_top_k_indices( np.nan_to_num(Rn[index_unlabeled]/Rs[index_unlabeled]), k)
				print(data.y.numpy().sum())
				recall= data.y.numpy()[indices].sum()/data.y.numpy().sum()
				accuracy= data.y.numpy()[indices].sum()/k
				print('Accuracy@'+str(k)+':', accuracy)
				print('Recall@'+str(k)+':', recall)
				Acc[k]=np.append(Acc[k],accuracy)
				Rec[k]=np.append(Rec[k],recall)
		elif args.model=="DOMINANT":
			t_total = time.time()	
			model = DOMINANT(epoch=args.epoch,hid_dim=args.hidden_dim,gpu=args.device,dropout=args.dropout,alpha=args.alpha,batch_size=0)
			model.fit(data)
			D = model.decision_function(data)
			D=np.nan_to_num(D)
			auc_score = eval_roc_auc(data.y.numpy(), D)
			NormSubSusp=np.append(NormSubSusp,auc_score)
			NormDivSusp=np.append(NormDivSusp,0)
			T=np.append(T,time.time() - t_total)
			for k in args.klist:
				print(k)
				indices = get_top_k_indices( D, k)
				print(data.y.numpy().sum())
				recall= data.y.numpy()[indices].sum()/data.y.numpy().sum()
				accuracy= data.y.numpy()[indices].sum()/k
				print('Accuracy@'+str(k)+':', accuracy)
				print('Recall@'+str(k)+':', recall)
				Acc[k]=np.append(Acc[k],accuracy)
				Rec[k]=np.append(Rec[k],recall)
		elif args.model=="ANOMALOUS":
			t_total = time.time()	
			model = ANOMALOUS(epoch=500)
			model.fit(data)
			D = model.decision_function(data)
			D=np.nan_to_num(D)
			auc_score = eval_roc_auc(data.y.numpy(), D)
			NormSubSusp=np.append(NormSubSusp,auc_score)
			NormDivSusp=np.append(NormDivSusp,0)
			T=np.append(T,time.time() - t_total)
			for k in args.klist:
				print(k)
				indices = get_top_k_indices( D, k)
				print(data.y.numpy().sum())
				recall= data.y.numpy()[indices].sum()/data.y.numpy().sum()
				accuracy= data.y.numpy()[indices].sum()/k
				print('Accuracy@'+str(k)+':', accuracy)
				print('Recall@'+str(k)+':', recall)
				Acc[k]=np.append(Acc[k],accuracy)
				Rec[k]=np.append(Rec[k],recall)

	
	write2=np.append(write2,str(round(np.mean(Norm)*100,1))+".pm."+str(round(np.std(Norm)*100,1)))
	write2=np.append(write2,str(round(np.mean(Susp)*100,1))+".pm."+str(round(np.std(Susp)*100,1)))	
	write2=np.append(write2,str(round(np.mean(NormSubSusp)*100,1))+".pm."+str(round(np.std(NormSubSusp)*100,1)))
	write2=np.append(write2,str(round(np.mean(NormDivSusp)*100,1))+".pm."+str(round(np.std(NormDivSusp)*100,1)))
	write2=np.append(write2,str(np.mean(T)))
	
	for k in args.klist:
		write2=np.append(write2,"Acc@"+str(k)+"; "+str(np.mean(Acc[k]))+".pm."+str(np.std(Acc[k])))
		#write2=np.append(write2,str(np.mean(Acc[k]))+".pm."+str(np.std(Acc[k])))
		write2=np.append(write2,"Rec@"+str(k)+"; "+str(np.mean(Rec[k]))+".pm."+str(np.std(Rec[k])))
		#write2=np.append(write2,str(np.mean(Rec[k]))+".pm."+str(np.std(Rec[k])))
 
	writer=csv.writer(F,dialect='excel',delimiter=';',lineterminator = '\n')
	writer.writerow(write2)	
