# -*- coding: utf-8 -*-
"""Deep Anomaly Detection on Attributed Networks (Norm)"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import NeighborLoader
from sklearn.utils.validation import check_is_fitted

from pygod.models.basic_nn import GCN
from pygod.models import BaseDetector
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn.conv import GATConv
from pygod.utils.utility import validate_device
from pygod.metrics import eval_roc_auc

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpointB.pt', trace_func=print):
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
            # self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
class Both(BaseDetector):
	"""
	DOMINANT (Deep Anomaly Detection on Attributed Networks)
	DOMINANT is an anomaly detector consisting of a shared graph
	convolutional encoder, a structure reconstruction decoder, and an
	attribute reconstruction decoder. The reconstruction mean square
	error of the decoders are defined as structure anomaly score and
	attribute anomaly score, respectively.

	See :cite:`ding2019deep` for details.

	Parameters
	----------
	hid_dim :  int, optional
		Hidden dimension of model. Default: ``0``.
	num_layers : int, optional
		Total number of layers in model. A half (ceil) of the layers
		are for the encoder, the other half (floor) of the layers are
		for decoders. Default: ``4``.
	dropout : float, optional
		Dropout rate. Default: ``0.``.
	weight_decay : float, optional
		Weight decay (L2 penalty). Default: ``0.``.
	act : callable activation function or None, optional
		Activation function if not None.
		Default: ``torch.nn.functional.relu``.
	alpha : float, optional
		Loss balance weight for attribute and structure.
		Default: ``0.5``.
	contamination : float, optional
		Valid in (0., 0.5). The proportion of outliers in the data set.
		Used when fitting to define the threshold on the decision
		function. Default: ``0.1``.
	lr : float, optional
		Learning rate. Default: ``0.004``.
	epoch : int, optional
		Maximum number of training epoch. Default: ``5``.
	gpu : int
		GPU Index, -1 for using CPU. Default: ``0``.
	batch_size : int, optional
		Minibatch size, 0 for full batch training. Default: ``0``.
	num_neigh : int, optional
		Number of neighbors in sampling, -1 for all neighbors.
		Default: ``-1``.
	verbose : bool
		Verbosity mode. Turn on to print out log information.
		Default: ``False``.

	Examples
	--------
	>>> from pygod.models import DOMINANT
	>>> model = DOMINANT()
	>>> model.fit(data) # PyG graph data object
	>>> prediction = model.predict(data)
	"""

	def __init__(self,
				 hid_dim=128,
				 num_layers=4,
				 dropout=0.1,
				 weight_decay=0.,
				 act=F.relu,
				 alpha=0.5,
				 contamination=0.1,
				 lr=5e-3,
				 epoch=100,
				 gpu=0,
				 batch_size=0,
				 num_neigh=-1,
				 verbose=False,
				 Sample_vs=torch.tensor([]),
				 Sample_vn=torch.tensor([]),
				 Sample_s=torch.tensor([]),
				 Sample_n=torch.tensor([])):
				 
		super(Both, self).__init__(contamination=contamination)

		# model param
		self.hid_dim = hid_dim
		self.num_layers = num_layers
		self.dropout = dropout
		self.weight_decay = weight_decay
		self.act = act
		self.alpha = alpha
		self.sample_n=Sample_n
		self.sample_vn=Sample_vn
		self.sample_s=Sample_s
		self.sample_vs=Sample_vs
		
		# training param
		self.lr = lr
		self.epoch = epoch
		self.device = validate_device(gpu)
		self.batch_size = batch_size
		self.num_neigh = num_neigh
		self.early_stopping = EarlyStopping(patience=100, verbose=True)
		# other param
		self.verbose = verbose
		self.model = None

	def fit(self, G, y_true=None):
		"""
		Fit detector with input data.

		Parameters
		----------
		G : torch_geometric.data.Data
			The input data.
		y_true : numpy.ndarray, optional
			The optional outlier ground truth labels used to monitor
			the training progress. They are not used to optimize the
			unsupervised model. Default: ``None``.

		Returns
		-------
		self : object
			Fitted estimator.
		"""
		G.node_idx = torch.arange(G.x.shape[0])
		G.s = to_dense_adj(G.edge_index)[0]
		if self.batch_size == 0:
			self.batch_size = G.x.shape[0]
		loader = NeighborLoader(G,
								[self.num_neigh] * self.num_layers,
								batch_size=self.batch_size)

		self.model = Both_Base(in_dim=G.x.shape[1],
								   hid_dim=self.hid_dim,
								   num_layers=self.num_layers,
								   dropout=self.dropout,
								   act=self.act).to(self.device)

		optimizer = torch.optim.Adam(self.model.parameters(),
									 lr=self.lr,
									 weight_decay=self.weight_decay)

		self.model.train()
		decision_scores = np.zeros(G.x.shape[0])
		train_losses = []
		val_losses = []
		for epoch in range(self.epoch):
			epoch_loss = 0
			for sampled_data in loader:
				batch_size = sampled_data.batch_size
				node_idx = sampled_data.node_idx
				x, s, edge_index = self.process_graph(sampled_data)

				x_, s_ = self.model(x, edge_index)
				score = self.loss_func(x[:batch_size],
									   x_[:batch_size],
									   s[:batch_size, node_idx],
									   s_[:batch_size])
				decision_scores[node_idx[:batch_size]] = score.detach() \
					.cpu().numpy()
				
				# loss = torch.mean(score[self.sample])
				l_s = torch.mean(score[self.sample_n])
				l_f = torch.mean(score[self.sample_s])
				loss_train=l_s/l_f
				
				# l_s = torch.mean(score[self.sample_vn])
				# l_f = torch.mean(score[self.sample_vs])
				# loss_val=l_s/l_f
				
				epoch_loss += loss_train.item() * batch_size

				optimizer.zero_grad()
				loss_train.backward()
				optimizer.step()
			train_losses.append(loss_train.detach().item())
			# val_losses.append(loss_val.detach().item())
			if self.verbose:
				print("Epoch {:04d}: Loss {:.4f}"
					  .format(epoch, epoch_loss / G.x.shape[0]), end='')
				if y_true is not None:
					auc = eval_roc_auc(y_true, decision_scores)
					print(" | AUC {:.4f}".format(auc), end='')
				print()
			
			train_losses.append(loss_train.detach().item())
			# val_losses.append(loss_val.detach().item())
			
			self.early_stopping(loss_train, self.model)
				
			# if self.early_stopping.early_stop:
				# print("Early stopping")
				# break
		self.decision_scores_ = decision_scores
		self._process_decision_scores()
		return self

	def decision_function(self, G):
		"""
		Predict raw anomaly score using the fitted detector. Outliers
		are assigned with larger anomaly scores.

		Parameters
		----------
		G : PyTorch Geometric Data instance (torch_geometric.data.Data)
			The input data.

		Returns
		-------
		outlier_scores : numpy.ndarray
			The anomaly score of shape :math:`N`.
		"""
		check_is_fitted(self, ['model'])
		G.node_idx = torch.arange(G.x.shape[0])
		G.s = to_dense_adj(G.edge_index)[0]

		loader = NeighborLoader(G,
								[self.num_neigh] * self.num_layers,
								batch_size=self.batch_size)
		
		self.model.load_state_dict(torch.load('checkpointB.pt'))
		self.model.eval()
		outlier_scores = np.zeros(G.x.shape[0])
		for sampled_data in loader:
			batch_size = sampled_data.batch_size
			node_idx = sampled_data.node_idx

			x, s, edge_index = self.process_graph(sampled_data)

			x_, s_ = self.model(x, edge_index)
			score = self.loss_func(x[:batch_size],
								   x_[:batch_size],
								   s[:batch_size, node_idx],
								   s_[:batch_size])

			outlier_scores[node_idx[:batch_size]] = score.detach() \
				.cpu().numpy()
		return outlier_scores

	def process_graph(self, G):
		"""
		Process the raw PyG data object into a tuple of sub data
		objects needed for the model.

		Parameters
		----------
		G : PyTorch Geometric Data instance (torch_geometric.data.Data)
			The input data.

		Returns
		-------
		x : torch.Tensor
			Attribute (feature) of nodes.
		s : torch.Tensor
			Adjacency matrix of the graph.
		edge_index : torch.Tensor
			Edge list of the graph.
		"""
		s = G.s.to(self.device)
		edge_index = G.edge_index.to(self.device)
		x = G.x.to(self.device)

		return x, s, edge_index

	def loss_func(self, x, x_, s, s_):
		# attribute reconstruction loss
		diff_attribute = torch.pow(x - x_, 2)
		attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))
		#attribute_errors = (attribute_errors-attribute_errors.min())/(attribute_errors.max()-attribute_errors.min())
		# structure reconstruction loss
		diff_structure = torch.pow(s - s_, 2)
		structure_errors = torch.sqrt(torch.sum(diff_structure, 1))
		#structure_errors=(structure_errors-structure_errors.min())/(structure_errors.max()-structure_errors.min())
			
		score = self.alpha * attribute_errors \
				+ (1 - self.alpha) * structure_errors
		return score


class Both_Base(nn.Module):
	def __init__(self,
				 in_dim,
				 hid_dim,
				 num_layers,
				 dropout,
				 act):
		super(Both_Base, self).__init__()

		# split the number of layers for the encoder and decoders
		decoder_layers = int(num_layers / 2)
		encoder_layers = num_layers - decoder_layers
		self.shared_encoder = GATConv(in_channels=in_dim,
								  out_channels=hid_dim,
								  dropout=dropout
								  )

		self.attr_decoder = GATConv(in_channels=hid_dim,
								out_channels=in_dim,
								dropout=dropout
								)

		self.struct_decoder = GATConv(in_channels=hid_dim,
								  out_channels=in_dim,
								  dropout=dropout
								  )

	def forward(self, x, edge_index):
		# encode
		h = self.shared_encoder(x, edge_index)
		# decode feature matrix
		x_ = self.attr_decoder(h, edge_index)
		# decode adjacency matrix
		h_ = self.struct_decoder(h, edge_index)
		s_ = h_ @ h_.T
		# return reconstructed matrices
		return x_, s_
