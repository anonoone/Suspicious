import torch
import numpy as np
from dataset import pyg_dataset, pyg_to_dgl
from models.baseline_model import Baseline

torch.manual_seed(21)
np.random.seed(2)

device = torch.device("cuda")
# model_list = ["iforest","lof", "hbos", "oc-svm", "bwgnn", "gin", "gat", "gcn"]
model_list = ["bwgnn", "gin", "gat", "gcn"]
# model_list = ["bwgnn"]
model_list = ["sage"]
# data_list = ["amazon_computer","cora","citeseer","amazon_photo","reddit","weibo","books"]
#data_list = ["amazon_photo"]
data_list = ["cora","citeseer","amazon_computer","amazon_photo"]
# data_list = ["pubmed"]
error=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#error=[0.8]
run_times = range(10)
hid_dim = 64
number_class = 2

for data_name in data_list:
	if data_name in ["pubmed","amazon_computer","citeseer","amazon_photo","weibo","books","reddit","flickr","cora"]:
		for model_name in model_list:
			for e in error:
				test_auc_list = []
				best_auc_list = []
				for run in run_times:
					data = pyg_dataset(dataset_name=data_name, dataset_spilt=[0.05,0.05,0.89], anomaly_type="min",error=e).dataset if model_name in ["iforest","lof", "hbos", "oc-svm"] else pyg_dataset(dataset_name=data_name, dataset_spilt=[0.05,0.05,0.89], anomaly_type="min",error=e).dataset.to(device)
					model = Baseline(model_name, data.x.shape[1], hid_dim, number_class, data, verbose=0) if model_name in ["iforest","lof", "hbos", "oc-svm"] else\
						Baseline(model_name, data.x.shape[1], hid_dim, number_class, data, verbose=0).to(device)
					test_auc,best_auc =	 model.fit()
					best_auc_list.append(best_auc)
					test_auc_list.append(test_auc)
				print (f"Baseline {model_name}; dataset {data_name}; anomaly type min; test auc mean {np.array(test_auc_list).mean()}; test auc std {np.array(test_auc_list).std()}; best val auc mean {np.array(best_auc_list).mean()}")
				record_string = f"Baseline {model_name}; dataset {data_name}; error {e};anomaly type min; test auc mean ${round(np.array(test_auc_list).mean()*100,1)}.pm.{round(np.array(test_auc_list).std()*100,1)}$\n"
				with open("result/baseline_for_all_dataset2.csv",'a') as f:
					f.write(record_string)
					
	if data_name in ["pubmed","amazon_computer","amazon_photo","citeseer","cora"]:
		for model_name in model_list:
			for e in error:
				test_auc_list = []
				best_auc_list = []
				for run in run_times:
					data = pyg_dataset(dataset_name=data_name, dataset_spilt=[0.05,0.05,0.89], anomaly_type="syn",error=e).dataset if model_name in ["iforest","lof", "hbos", "oc-svm"] else pyg_dataset(dataset_name=data_name, dataset_spilt=[0.05,0.05,0.89], anomaly_type="syn",error=e).dataset.to(device)
					model = Baseline(model_name, data.x.shape[1], hid_dim, number_class, data, verbose=0) if model_name in ["iforest","lof", "hbos", "oc-svm"] else\
						Baseline(model_name, data.x.shape[1], hid_dim, number_class, data, verbose=0).to(device)
					test_auc,best_auc =	 model.fit()
					best_auc_list.append(best_auc)
					test_auc_list.append(test_auc)
				print (f"Baseline {model_name}; dataset {data_name}; anomaly type syn; test auc mean {np.array(test_auc_list).mean()}; test auc std {np.array(test_auc_list).std()}; best val auc mean {np.array(best_auc_list).mean()}")
				record_string = f"Baseline {model_name}; dataset {data_name}; error {e};anomaly type syn; test auc mean ${round(np.array(test_auc_list).mean()*100,1)}.pm.{round(np.array(test_auc_list).std()*100,1)}$\n"
				with open("result/baseline_for_all_dataset2.csv",'a') as f:
					f.write(record_string) 
